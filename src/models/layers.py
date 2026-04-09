"""
通用神经网络层模块。

收纳本项目中跨模型复用的基础层，例如 TD-MPC2 风格的
``SimNorm`` 单纯形归一化层与带 ``LayerNorm`` + 激活的
``NormedLinear`` 线性层。其它模型文件（编码器、动力学、
策略、价值等）应统一从这里导入，避免重复定义。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SimNorm(nn.Module):
    """
    单纯形归一化层（Simplicial Normalization）。

    将最后一维切成若干等长子向量，分别做 ``softmax``，再展平回原形。
    源自 TD-MPC2，论文: https://arxiv.org/abs/2204.00616 。
    """

    def __init__(
        self,
        group_dim: int,
    ) -> None:
        """
        参数:
            group_dim: 每个单纯形组的维度，调用方需保证最后一维可整除。
        """
        super().__init__()
        self.group_dim = group_dim

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """对最后一维分组并执行 ``softmax`` 归一化。"""
        original_shape = inputs.shape
        grouped = inputs.view(*original_shape[:-1], -1, self.group_dim)
        normalized = F.softmax(grouped, dim=-1)
        return normalized.view(*original_shape)

    def extra_repr(self) -> str:
        """打印额外的层信息。"""
        return f"group_dim={self.group_dim}"


class NormedLinear(nn.Linear):
    """
    带 ``LayerNorm`` 与激活函数的线性层。

    顺序为 ``Linear -> LayerNorm -> Activation``，默认激活为 ``Mish``。
    与 TD-MPC2 中 ``NormedLinear`` 的语义保持一致。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        """
        参数:
            in_features: 输入维度。
            out_features: 输出维度。
            activation: 激活函数模块；为 ``None`` 时使用 ``nn.Mish``。
            bias: 是否启用偏置项。
        """
        super().__init__(in_features, out_features, bias=bias)
        self.layer_norm = nn.LayerNorm(out_features)
        self.activation = activation if activation is not None else nn.Mish()

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """执行 ``Linear -> LayerNorm -> Activation`` 前向。"""
        projected = super().forward(inputs)
        return self.activation(self.layer_norm(projected))
