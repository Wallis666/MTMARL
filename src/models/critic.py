"""
多智能体多任务价值网络（critic）。

参考 m3w-marl ``WorldModelCritic`` / ``DisRegQNet`` 的设计：

* 中心化价值估计（CTDE）：把所有智能体的潜在 ``z`` 与联合动作
  ``a`` 沿智能体维度展平后拼接，作为单个 MLP 的输入。
* **two-hot 离散回归头**：输出 ``num_bins`` 个 logits，由训练侧
  通过 ``TwoHotProcessor`` 解码为标量 Q 值。这种做法源自
  TD-MPC2 / DreamerV3，比直接回归标量更稳定。
* **Twin Q（双 Q 网络）**：内部维护两套独立的 Q 网络，``forward``
  同时返回 ``(q1_logits, q2_logits)``，训练时用最小值估计抑制
  价值高估，与 SAC / TD3 / m3w 一致。

关键约定:

* 输入张量形状: ``latents`` 为 ``(B, n_agents, latent_dim)``，
  ``actions`` 为 ``(B, n_agents, action_dim)``，与 actor / dynamics /
  reward 对齐。
* 任务条件化已由 ``MultiTaskMaMuJoCo`` 注入到 ``z``，所有任务共享
  同一套 critic 权重，无需在外部再拼任务嵌入。
* 输出 logits 形状均为 ``(B, num_bins)``，``num_bins=1`` 时退化为
  对称 log 标量预测（与 m3w / TD-MPC2 的 ``TwoHotProcessor`` 兼容）。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.layers import build_normed_mlp


class DistributionalQNetwork(nn.Module):
    """
    单个中心化离散回归 Q 网络。

    把 ``(B, n_agents, latent_dim)`` 的潜在与 ``(B, n_agents,
    action_dim)`` 的联合动作沿最后一维拼接后展平，再通过 MLP 输出
    ``num_bins`` 个 logits 供 two-hot 解码。
    """

    def __init__(
        self,
        n_agents: int,
        latent_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512),
        num_bins: int = 101,
    ) -> None:
        """
        参数:
            n_agents: 智能体数量，对应 ``MultiTaskMaMuJoCo.max_n_agents``。
            latent_dim: 单个智能体的潜在维度，与 ``ObsEncoder.latent_dim`` 对齐。
            action_dim: 单个智能体的动作维度，对应 ``max_action_dim``。
            hidden_dims: 主干 MLP 的隐藏层宽度序列。
            num_bins: two-hot 离散回归的 bin 数；``1`` 时退化为标量回归。
        """
        super().__init__()
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.input_dim = n_agents * (latent_dim + action_dim)

        self.trunk = build_normed_mlp(
            in_dim=self.input_dim,
            hidden_dims=hidden_dims,
            out_dim=num_bins,
            output_activation=None,
        )
        # 与 m3w 保持一致：把最后一层权重清零，初始 Q 估计偏向中性
        last_linear = self.trunk[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            latents: 形状 ``(B, n_agents, latent_dim)`` 的潜在张量。
            actions: 形状 ``(B, n_agents, action_dim)`` 的联合动作张量。

        返回:
            形状 ``(B, num_bins)`` 的 Q 值 logits。
        """
        batch_size = latents.shape[0]
        joint_input = torch.cat([latents, actions], dim=-1)
        flattened = joint_input.reshape(batch_size, self.input_dim)
        return self.trunk(flattened)


class TwinQCritic(nn.Module):
    """
    Twin Q 中心化 critic（CTDE）。

    内部维护两套独立的 :class:`DistributionalQNetwork`，``forward`` 返回
    ``(q1_logits, q2_logits)``。训练侧把它们解码为标量后取较小值，用于
    抑制价值高估；外部 trainer 负责管理目标网络与 Polyak 软更新。
    """

    def __init__(
        self,
        n_agents: int,
        latent_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512),
        num_bins: int = 101,
    ) -> None:
        """
        参数:
            n_agents: 智能体数量，对应 ``MultiTaskMaMuJoCo.max_n_agents``。
            latent_dim: 单个智能体的潜在维度，与 ``ObsEncoder.latent_dim`` 对齐。
            action_dim: 单个智能体的动作维度，对应 ``max_action_dim``。
            hidden_dims: 单个 Q 网络主干 MLP 的隐藏层宽度序列。
            num_bins: two-hot 离散回归的 bin 数。
        """
        super().__init__()
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_bins = num_bins

        self.q1 = DistributionalQNetwork(
            n_agents=n_agents,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_bins=num_bins,
        )
        self.q2 = DistributionalQNetwork(
            n_agents=n_agents,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_bins=num_bins,
        )

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        同时输出两个 Q 网络的 logits。

        参数:
            latents: 形状 ``(B, n_agents, latent_dim)`` 的潜在张量。
            actions: 形状 ``(B, n_agents, action_dim)`` 的联合动作张量。

        返回:
            ``(q1_logits, q2_logits)``，形状均为 ``(B, num_bins)``。

        异常:
            ValueError: 当输入张量维度或形状不符合约定时抛出。
        """
        self._validate_inputs(latents, actions)
        return self.q1(latents, actions), self.q2(latents, actions)

    def q_min(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回两个 Q 网络 logits 的逐元素最小值，用于目标价值估计。

        参数:
            latents: 形状 ``(B, n_agents, latent_dim)`` 的潜在张量。
            actions: 形状 ``(B, n_agents, action_dim)`` 的联合动作张量。

        返回:
            ``(q1_logits, q2_logits)``，与 :meth:`forward` 一致；调用方
            通常在外部用 ``TwoHotProcessor.logits_decode_scalar`` 解码后
            再 ``torch.min`` 得到目标 Q 值。

        说明:
            这里**不直接对 logits 取最小值**，因为 logits 是概率分布而
            非已解码的标量；最小值操作必须发生在标量域。本方法只是
            语义上的 alias，方便调用方表达"我要双 Q 的两个估计"。
        """
        return self.forward(latents, actions)

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
