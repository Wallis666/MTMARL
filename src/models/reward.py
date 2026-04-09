"""
多智能体多任务奖励模型。

参考 m3w-marl ``CenMoERewardModel`` 的设计，把所有智能体的潜在
``z_t`` 与联合动作 ``a_t`` 拼成 token 序列后，经过 **Noisy Top-K
门控** 把每个样本路由到 ``k`` 个 **自注意力专家**，再用一个共享的
回归头预测团队奖励的 two-hot logits（与 TD-MPC2 的回归方式一致）。

整体流程::

    tokens = concat(z, a)                       # [B, N_a, D]
    flat   = tokens.reshape(B, -1)              # [B, N_a*D]
    gates, load, logits, aux = router(flat)     # 噪声 Top-K 路由
    for e in experts:
        if 该样本被路由到 e:
            y[mask] += gates[mask, e] * e(tokens[mask])
    reward_logits = reward_head(y.reshape(B, -1))  # [B, num_bins]

约定:

* 任务 one-hot 已由 ``MultiTaskMaMuJoCo`` 写入观测前缀，并通过
  ``ObsEncoder`` 进入 ``z_t``，因此本模块不需要再显式注入任务嵌入。
* 输出 ``reward_logits`` 形状 ``(B, num_bins)``，给上层做 two-hot
  解码（``num_bins=1`` 时即为对称 log 后的标量预测）。
* ``forward`` 同时返回 ``aux``，包含负载均衡损失等供训练侧加权。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F


class SelfAttentionExpert(nn.Module):
    """
    自注意力专家模块。

    在 ``(B, N_a, D)`` 的智能体 token 序列上做一次多头自注意力 +
    前馈网络（带残差与 LayerNorm），让所有智能体相互交互后再返回
    同形状的输出，作为 MoE 路由后的局部计算单元。
    """

    def __init__(
        self,
        token_dim: int,
        n_heads: int = 1,
        ffn_hidden_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        """
        参数:
            token_dim: 单个 token 的维度（``latent_dim + action_dim``）。
            n_heads: 自注意力头数，需整除 ``token_dim``。
            ffn_hidden_dim: 前馈网络隐藏层宽度。
            dropout: 注意力与前馈中的 dropout 概率。
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, token_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            tokens: 形状 ``(B, N_a, token_dim)`` 的智能体 token 序列。

        返回:
            形状与输入相同的张量。
        """
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        tokens = self.attention_norm(tokens + attended)
        tokens = self.feed_forward_norm(tokens + self.feed_forward(tokens))
        return tokens


class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K 门控路由器。

    源自 *Outrageously Large Neural Networks* (Shazeer 等,
    https://arxiv.org/abs/1701.06538) 的稀疏 MoE 路由：训练阶段在
    logits 上加入可学习方差的高斯噪声并做 top-k 选择，再 softmax
    归一化得到 ``gates``；同时输出负载均衡相关统计量供训练侧使用。
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        top_k: int = 2,
        use_noisy_gating: bool = True,
    ) -> None:
        """
        参数:
            in_dim: 输入特征维度（已展平的 token 联合特征）。
            n_experts: 专家总数。
            top_k: 每个样本选取的专家数，需 ``<= n_experts``。
            use_noisy_gating: 是否在训练时启用噪声门控。

        异常:
            ValueError: 当 ``top_k`` 大于 ``n_experts`` 时抛出。
        """
        super().__init__()
        if top_k > n_experts:
            raise ValueError(
                f"top_k ({top_k}) 不能大于 n_experts ({n_experts})。"
            )

        self.in_dim = in_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_noisy_gating = use_noisy_gating

        self.gate_weight = nn.Parameter(torch.zeros(in_dim, n_experts))
        self.noise_weight = nn.Parameter(torch.zeros(in_dim, n_experts))
        self.softplus = nn.Softplus()

        self.register_buffer("normal_mean", torch.tensor([0.0]))
        self.register_buffer("normal_std", torch.tensor([1.0]))

    @staticmethod
    def _coefficient_of_variation_squared(
        values: torch.Tensor,
    ) -> torch.Tensor:
        """计算变异系数平方，用于负载均衡损失。"""
        eps = 1e-10
        if values.shape[0] == 1:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return values.float().var() / (values.float().mean() ** 2 + eps)

    @staticmethod
    def _z_loss(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Router z-loss，鼓励 logits 数值稳定。"""
        return torch.log(torch.exp(logits).sum(-1)).mean()

    @staticmethod
    def _gates_to_load(
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """统计每个专家被命中的样本数（硬负载）。"""
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self,
        clean_logits: torch.Tensor,
        noisy_logits: torch.Tensor,
        noise_stddev: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        估计在加噪情况下每个专家进入 top-k 的概率，用于平滑负载估计。
        与 Shazeer 2017 原论文实现保持一致。
        """
        batch_size = clean_logits.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        device = clean_logits.device
        in_positions = torch.arange(batch_size, device=device) * m + self.top_k
        in_threshold = torch.unsqueeze(
            torch.gather(top_values_flat, 0, in_positions),
            1,
        )
        is_in = torch.gt(noisy_logits, in_threshold)
        out_positions = in_positions - 1
        out_threshold = torch.unsqueeze(
            torch.gather(top_values_flat, 0, out_positions),
            1,
        )

        normal = Normal(self.normal_mean, self.normal_std)
        prob_if_in = normal.cdf((clean_logits - in_threshold) / noise_stddev)
        prob_if_out = normal.cdf((clean_logits - out_threshold) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def forward(
        self,
        flat_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """
        参数:
            flat_tokens: 形状 ``(B, in_dim)`` 的展平 token 联合特征。

        返回:
            ``(gates, load, logits, aux)``，其中:

            * ``gates`` 形状 ``(B, n_experts)``，稀疏门控权重；
            * ``load`` 形状 ``(n_experts,)``，每个专家的负载估计；
            * ``logits`` 形状 ``(B, n_experts)``，路由 logits；
            * ``aux`` 含负载均衡损失等训练侧需要的标量。
        """
        clean_logits = flat_tokens @ self.gate_weight
        if self.use_noisy_gating and self.training:
            raw_noise_stddev = flat_tokens @ self.noise_weight
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            noise_stddev = None
            noisy_logits = None
            logits = clean_logits

        top_values, top_indices = logits.topk(
            min(self.top_k + 1, self.n_experts),
            dim=1,
        )
        top_k_values = top_values[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = F.softmax(top_k_values, dim=1)

        gates = torch.zeros_like(logits).scatter(1, top_k_indices, top_k_gates)

        if (
            self.use_noisy_gating
            and self.top_k < self.n_experts
            and self.training
        ):
            load = self._prob_in_top_k(
                clean_logits,
                noisy_logits,
                noise_stddev,
                top_values,
            ).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        load_balancing_loss = (
            self._coefficient_of_variation_squared(importance)
            + self._coefficient_of_variation_squared(load)
            + self._z_loss(logits)
        )

        aux = {
            "load_balancing_loss": load_balancing_loss,
            "importance": importance,
            "load": load,
            "logits": logits,
        }
        return gates, load, logits, aux


class SparseMoERewardModel(nn.Module):
    """
    基于 Noisy Top-K MoE 与自注意力专家的中心化奖励模型。

    输入为所有智能体的潜在 ``z`` 与联合动作 ``a``，输出整支团队的
    奖励 logits（用于 two-hot 离散回归）。任务条件化已由
    ``MultiTaskMaMuJoCo`` 注入到 ``z`` 之中，因此本模块对所有任务
    共享同一套权重。
    """

    def __init__(
        self,
        n_agents: int,
        latent_dim: int,
        action_dim: int,
        n_experts: int = 4,
        top_k: int = 2,
        num_bins: int = 101,
        n_attention_heads: int = 1,
        expert_ffn_hidden_dim: int = 1024,
        expert_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        use_noisy_gating: bool = True,
    ) -> None:
        """
        参数:
            n_agents: 智能体数量，对应 ``MultiTaskMaMuJoCo.max_n_agents``。
            latent_dim: 单个智能体的潜在维度，与 ``ObsEncoder.latent_dim`` 对齐。
            action_dim: 单个智能体的动作维度，对应 ``max_action_dim``。
            n_experts: 专家数量。
            top_k: 每个样本激活的专家数。
            num_bins: 团队奖励 two-hot 回归的 bin 数（``1`` 表示退化为标量）。
            n_attention_heads: 自注意力专家的多头数，需整除 ``token_dim``。
            expert_ffn_hidden_dim: 专家前馈网络隐藏层宽度。
            expert_dropout: 专家内部 dropout 概率。
            head_hidden_dim: 奖励回归头的隐藏层宽度。
            use_noisy_gating: 是否启用 Noisy Top-K 门控。

        异常:
            ValueError: 当 ``token_dim`` 无法被 ``n_attention_heads`` 整除时抛出。
        """
        super().__init__()

        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.token_dim = latent_dim + action_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.num_bins = num_bins

        if self.token_dim % n_attention_heads != 0:
            raise ValueError(
                f"token_dim ({self.token_dim}) 必须能被 n_attention_heads "
                f"({n_attention_heads}) 整除。"
            )

        self.router = NoisyTopKRouter(
            in_dim=n_agents * self.token_dim,
            n_experts=n_experts,
            top_k=top_k,
            use_noisy_gating=use_noisy_gating,
        )

        self.experts = nn.ModuleList(
            [
                SelfAttentionExpert(
                    token_dim=self.token_dim,
                    n_heads=n_attention_heads,
                    ffn_hidden_dim=expert_ffn_hidden_dim,
                    dropout=expert_dropout,
                )
                for _ in range(n_experts)
            ],
        )

        self.reward_head = nn.Sequential(
            nn.Linear(n_agents * self.token_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_bins),
        )

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        预测团队奖励 logits。

        参数:
            latents: 形状 ``(B, n_agents, latent_dim)`` 的当前潜在。
            actions: 形状 ``(B, n_agents, action_dim)`` 的联合动作。

        返回:
            ``(reward_logits, aux)``，其中 ``reward_logits`` 形状
            ``(B, num_bins)``；``aux`` 包含 ``load_balancing_loss``、
            ``gates``、``logits`` 等训练侧需要的字段。

        异常:
            ValueError: 当输入张量维度或形状不符合约定时抛出。
        """
        self._validate_inputs(latents, actions)

        batch_size = latents.shape[0]
        tokens = torch.cat([latents, actions], dim=-1)
        flat_tokens = tokens.reshape(batch_size, -1)

        gates, _, _, router_aux = self.router(flat_tokens)

        mixed_tokens = torch.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            mask = gates[:, expert_index] > 0
            if not mask.any():
                continue
            expert_output = expert(tokens[mask])
            weight = gates[mask, expert_index].view(-1, 1, 1)
            mixed_tokens[mask] = mixed_tokens[mask] + weight * expert_output

        reward_logits = self.reward_head(mixed_tokens.reshape(batch_size, -1))

        aux = {
            "load_balancing_loss": router_aux["load_balancing_loss"],
            "gates": gates,
            "logits": router_aux["logits"],
        }
        return reward_logits, aux

    def predict(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """仅返回奖励 logits 的便捷接口（供 rollout / 推理使用）。"""
        reward_logits, _ = self.forward(latents, actions)
        return reward_logits

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        """对 ``forward`` 的输入做形状校验。"""
        if latents.dim() != 3 or actions.dim() != 3:
            raise ValueError(
                f"latents/actions 必须是 3 维张量 (B, n_agents, dim)，"
                f"实际得到 {latents.shape} 与 {actions.shape}。"
            )
        if latents.shape[0] != actions.shape[0]:
            raise ValueError(
                f"latents 与 actions 的 batch 维度不一致："
                f"{latents.shape[0]} vs {actions.shape[0]}。"
            )
        if (
            latents.shape[1] != self.n_agents
            or actions.shape[1] != self.n_agents
        ):
            raise ValueError(
                f"智能体维度与构造时的 n_agents={self.n_agents} 不一致："
                f"latents={latents.shape[1]}, actions={actions.shape[1]}。"
            )
        if latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latents 末维 {latents.shape[-1]} 与 latent_dim "
                f"{self.latent_dim} 不一致。"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions 末维 {actions.shape[-1]} 与 action_dim "
                f"{self.action_dim} 不一致。"
            )
