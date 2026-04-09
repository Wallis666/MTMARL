"""
观测编码器模块。

把 ``MultiTaskMaMuJoCo`` 给出的、已经在最前面拼接了任务 one-hot 的
对齐观测向量编码到固定维度的潜空间，供后续策略 / 值函数 / 世界模型
使用。整体设计参考 TD-MPC2 的状态编码器（``mlp + LayerNorm + Mish +
SimNorm``），并适配多智能体场景：每个智能体共享同一个 MLP 主干，
batch 维度自然包含 ``num_envs * n_agents``。

输入张量形状约定:

* ``observations``: ``(..., n_agents, obs_dim)`` 或 ``(..., obs_dim)``，
  其中 ``obs_dim`` 已经包含 ``MultiTaskMaMuJoCo`` 写入的 ``n_total_tasks``
  长度的任务 one-hot 前缀。
* 输出 latent: ``(..., latent_dim)``，最后一维由 ``SimNorm`` 做单纯形
  归一化，便于作为世界模型的潜在状态。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.layers import NormedLinear, SimNorm


class ObsEncoder(nn.Module):
    """
    多智能体共享观测编码器。

    将形如 ``(..., obs_dim)`` 的对齐观测向量编码到 ``latent_dim`` 维潜
    空间。所有智能体共享同一套权重，便于参数高效与跨任务迁移；任务
    one-hot 已经由 ``MultiTaskMaMuJoCo`` 写在 ``obs_dim`` 的最前面，
    因此本模块不需要再单独传入任务嵌入。
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 512,
        hidden_dims: Sequence[int] = (256, 256),
        simnorm_group_dim: int = 8,
    ) -> None:
        """
        参数:
            obs_dim: 单个智能体的对齐观测维度（含任务 one-hot 前缀）。
            latent_dim: 输出潜空间维度，必须能被 ``simnorm_group_dim`` 整除。
            hidden_dims: 主干 MLP 的各隐藏层宽度。
            simnorm_group_dim: ``SimNorm`` 单纯形分组维度。

        异常:
            ValueError: 当 ``latent_dim`` 无法被 ``simnorm_group_dim`` 整除时抛出。
        """
        super().__init__()
        if latent_dim % simnorm_group_dim != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) 必须能被 "
                f"simnorm_group_dim ({simnorm_group_dim}) 整除。"
            )

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        layers: list[nn.Module] = []
        previous_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(NormedLinear(previous_dim, hidden_dim))
            previous_dim = hidden_dim
        layers.append(
            NormedLinear(
                previous_dim,
                latent_dim,
                activation=SimNorm(simnorm_group_dim),
            ),
        )
        self.trunk = nn.Sequential(*layers)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向计算。

        参数:
            observations: 任意前缀维度 + 末维 ``obs_dim`` 的浮点张量。
                典型形状为 ``(num_envs, n_agents, obs_dim)``。

        返回:
            末维替换为 ``latent_dim`` 的潜空间张量。

        异常:
            ValueError: 当输入末维与 ``obs_dim`` 不一致时抛出。
        """
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(
                f"输入观测末维 {observations.shape[-1]} 与编码器约定的 "
                f"obs_dim {self.obs_dim} 不一致。"
            )
        return self.trunk(observations)
