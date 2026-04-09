"""
多智能体多任务潜在动力学模型（Soft MoE 版）。

本模块提供 :class:`SoftMoEDynamics`，参考 m3w-marl 中的
``CenMoEDynamicsModel`` 设计，把多个 MLP 专家通过 **Soft Mixture of
Experts**（Puigcerver 等, https://arxiv.org/abs/2308.00951）的方式
软路由起来，对联合 ``(z_t, a_t)`` 进行建模并预测下一时刻的潜在状态
``z_{t+1}``。相较于纯中心化 MLP，Soft MoE 能在不同任务 / 不同智能体
组合下激活不同专家，更适合 MTMARL 这种**跨任务、跨身体形态**的统一
世界模型场景。

总体流程（与 m3w 一致，但变量命名更清晰）::

    x = concat(z_t, a_t)                          # [B, N_a, d_token]
    routing_logits = einsum("bnd,des->bnes", x, phi)
    dispatch_weights = softmax(routing_logits, dim=智能体维)
    slot_inputs = einsum("bnes,bnd->besd", dispatch_weights, x)
    slot_outputs = expert_e(slot_inputs[:, e])    # 每个专家独立处理自己的 slot
    combine_weights = softmax(routing_logits.flatten(experts*slots), -1)
    z_{t+1} = einsum("bnz,bzd->bnd", combine_weights, slot_outputs)

约定:

* 任务 one-hot 已由 ``MultiTaskMaMuJoCo`` 写入观测前缀，并通过
  ``ObsEncoder`` 进入 ``z_t``，因此本模块**不需要**再显式注入任务嵌入。
* 输入/输出张量形状始终为 ``(B, n_agents, latent_dim)``，与
  ``ObsEncoder`` 的输出 / actor 的输入对齐。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.layers import SimNorm, build_normed_mlp


class SoftMoEDynamics(nn.Module):
    """
    基于 Soft MoE 的多智能体潜在动力学模型。

    每个专家是一个独立的 MLP，把 ``(z, a)`` 映射到下一时刻潜在 ``z'``。
    路由器为每对 ``(智能体 token, 专家 slot)`` 生成软权重，并按
    Soft MoE 的方式做 dispatch / combine，使得每个智能体的下一时刻
    潜在都是所有专家 slot 输出的加权组合。
    """

    def __init__(
        self,
        n_agents: int,
        latent_dim: int,
        action_dim: int,
        n_experts: int = 4,
        n_slots_per_expert: int = 1,
        hidden_dims: Sequence[int] = (512, 512),
        simnorm_group_dim: int = 8,
    ) -> None:
        """
        参数:
            n_agents: 智能体数量，对应 ``MultiTaskMaMuJoCo.max_n_agents``。
            latent_dim: 单个智能体的潜在维度，与 ``ObsEncoder.latent_dim`` 对齐。
            action_dim: 单个智能体的动作维度，对应 ``max_action_dim``。
            n_experts: 专家数量。
            n_slots_per_expert: 每个专家的 slot 数（Soft MoE 中的 ``S``）。
            hidden_dims: 单个专家 MLP 的隐藏层宽度序列。
            simnorm_group_dim: 末层 ``SimNorm`` 的分组维度，需整除 ``latent_dim``。

        异常:
            ValueError: 当 ``latent_dim`` 无法被 ``simnorm_group_dim`` 整除，
                或 ``n_experts`` / ``n_slots_per_expert`` 不为正时抛出。
        """
        super().__init__()
        if latent_dim % simnorm_group_dim != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) 必须能被 "
                f"simnorm_group_dim ({simnorm_group_dim}) 整除。"
            )
        if n_experts <= 0 or n_slots_per_expert <= 0:
            raise ValueError(
                f"n_experts 与 n_slots_per_expert 必须为正整数，"
                f"实际得到 n_experts={n_experts}, "
                f"n_slots_per_expert={n_slots_per_expert}。"
            )

        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.token_dim = latent_dim + action_dim
        self.n_experts = n_experts
        self.n_slots_per_expert = n_slots_per_expert

        # 路由参数 phi：[token_dim, n_experts, n_slots_per_expert]
        # 与 m3w 完全一致的初始化方式
        scale = 1.0 / math.sqrt(self.token_dim)
        self.routing_phi = nn.Parameter(
            torch.randn(
                self.token_dim,
                n_experts,
                n_slots_per_expert,
            ) * scale,
        )

        # 每个专家是独立的 MLP，输出维度 = latent_dim，末层使用 SimNorm
        self.experts = nn.ModuleList(
            [
                build_normed_mlp(
                    in_dim=self.token_dim,
                    hidden_dims=hidden_dims,
                    out_dim=latent_dim,
                    output_activation=SimNorm(simnorm_group_dim),
                )
                for _ in range(n_experts)
            ],
        )

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测下一时刻的联合潜在状态。

        参数:
            latents: 形状 ``(B, n_agents, latent_dim)`` 的当前潜在。
            actions: 形状 ``(B, n_agents, action_dim)`` 的联合动作。

        返回:
            形状 ``(B, n_agents, latent_dim)`` 的下一时刻潜在张量。

        异常:
            ValueError: 当输入张量维度或形状不符合约定时抛出。
        """
        self._validate_inputs(latents, actions)

        batch_size = latents.shape[0]

        # 1) 拼接成 token：[B, N_a, token_dim]
        tokens = torch.cat([latents, actions], dim=-1)

        # 2) 路由打分：[B, N_a, N_e, S]
        routing_logits = torch.einsum(
            "bnd,des->bnes",
            tokens,
            self.routing_phi,
        )

        # 3) Dispatch：在“智能体 token”维度上做 softmax，每个 slot 收到的
        #    输入是所有智能体 token 的加权和
        dispatch_weights = F.softmax(routing_logits, dim=1)  # [B, N_a, N_e, S]
        slot_inputs = torch.einsum(
            "bnes,bnd->besd",
            dispatch_weights,
            tokens,
        )  # [B, N_e, S, token_dim]

        # 4) 每个专家处理自己的 slot：得到 [B, N_e, S, latent_dim]
        expert_outputs_list: list[torch.Tensor] = []
        for expert_index, expert in enumerate(self.experts):
            # slot_inputs[:, expert_index] -> [B, S, token_dim]
            expert_outputs_list.append(expert(slot_inputs[:, expert_index]))
        slot_outputs = torch.stack(
            expert_outputs_list,
            dim=1,
        )  # [B, N_e, S, latent_dim]

        # 5) Combine：把 (N_e, S) 展平后在最后一维做 softmax，
        #    再用它把 slot 输出加权回每个智能体
        combine_weights = routing_logits.reshape(
            batch_size,
            self.n_agents,
            self.n_experts * self.n_slots_per_expert,
        )
        combine_weights = F.softmax(combine_weights, dim=-1)

        slot_outputs_flat = slot_outputs.reshape(
            batch_size,
            self.n_experts * self.n_slots_per_expert,
            self.latent_dim,
        )

        next_latents = torch.einsum(
            "bnz,bzd->bnd",
            combine_weights,
            slot_outputs_flat,
        )  # [B, N_a, latent_dim]

        return next_latents

    def rollout(
        self,
        initial_latents: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        在潜空间中按给定动作序列展开多步预测。

        参数:
            initial_latents: 形状 ``(B, n_agents, latent_dim)`` 的初始潜在。
            action_sequence: 形状 ``(T, B, n_agents, action_dim)`` 的动作序列。

        返回:
            形状 ``(T, B, n_agents, latent_dim)`` 的逐步预测潜在轨迹，
            其中第 ``t`` 项表示执行 ``action_sequence[t]`` 之后的潜在状态。

        异常:
            ValueError: 当 ``action_sequence`` 维度不符合约定时抛出。
        """
        if action_sequence.dim() != 4:
            raise ValueError(
                f"action_sequence 必须是 4 维张量 (T, B, n_agents, action_dim)，"
                f"实际得到 {action_sequence.shape}。"
            )

        horizon = action_sequence.shape[0]
        current = initial_latents
        predicted_latents: list[torch.Tensor] = []
        for step_index in range(horizon):
            current = self.forward(current, action_sequence[step_index])
            predicted_latents.append(current)
        return torch.stack(predicted_latents, dim=0)

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
