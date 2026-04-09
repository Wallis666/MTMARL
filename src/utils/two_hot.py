"""
Two-hot 离散回归编解码工具。

参考 m3w-marl ``world_models.TwoHotProcessor`` 与 TD-MPC2 / DreamerV3 的
做法：把一个连续标量目标 ``y`` 通过对称对数压缩 (`symlog`) 后映射到
``[vmin, vmax]`` 上等距划分的 ``num_bins`` 个 bin，再在相邻两个 bin 上
分配权重得到 **two-hot 向量**；模型输出 ``num_bins`` 个 logits，用
**交叉熵 / KL** 拟合该 two-hot 目标，预测时再用 ``softmax`` 加权回
``symexp`` 解码得到原标量。

这种"对标量做 two-hot 离散回归"的目标比直接 MSE 拟合更稳定，对动态
范围大的奖励 / Q 值尤其有效，是 ``SparseMoERewardModel`` 与
``TwinQCritic`` 的训练目标编码方式。

退化情形:

* ``num_bins == 0``: 不做编码，直接走 MSE / 标量；
* ``num_bins == 1``: 不做 two-hot，仅做 ``symlog`` 标量回归；
* ``num_bins >= 2``: 标准 two-hot 离散回归。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def symlog(
    value: torch.Tensor,
) -> torch.Tensor:
    """
    对称 log 压缩 ``sign(x) * log(1 + |x|)``。

    参数:
        value: 任意形状的张量。

    返回:
        与输入同形状、动态范围被对数压缩后的张量。
    """
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(
    value: torch.Tensor,
) -> torch.Tensor:
    """
    :func:`symlog` 的逆变换 ``sign(x) * (exp(|x|) - 1)``。

    参数:
        value: 任意形状的张量。

    返回:
        与输入同形状的解压缩张量。
    """
    return torch.sign(value) * torch.expm1(torch.abs(value))


class TwoHotProcessor:
    """
    Two-hot 离散回归编解码器。

    构造时确定 bin 数与值域，之后即可在张量上做 ``encode`` /
    ``decode`` / ``cross_entropy_loss``。本类不持有可学习参数，
    但需要在合适的 ``device`` 上保存 bin 中心张量，因此提供
    :meth:`to` 在外部模型迁移设备时同步迁移。
    """

    def __init__(
        self,
        num_bins: int,
        vmin: float,
        vmax: float,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        参数:
            num_bins: 离散 bin 数。``0`` 表示纯标量回归，``1`` 表示仅
                做 ``symlog`` 压缩，``>=2`` 启用 two-hot 离散回归。
            vmin: ``symlog`` 后允许的最小值，所有目标会被裁剪到该下界。
            vmax: ``symlog`` 后允许的最大值，所有目标会被裁剪到该上界。
            device: bin 中心张量所在的设备。

        异常:
            ValueError: 当 ``num_bins`` 为负或 ``vmin >= vmax`` 时抛出。
        """
        if num_bins < 0:
            raise ValueError(f"num_bins 必须 >= 0，实际得到 {num_bins}。")
        if num_bins >= 2 and vmin >= vmax:
            raise ValueError(
                f"vmin ({vmin}) 必须严格小于 vmax ({vmax})。"
            )

        self.num_bins = num_bins
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.device = torch.device(device)

        if num_bins >= 2:
            self.bin_size = (self.vmax - self.vmin) / (num_bins - 1)
            self.bin_centers = torch.linspace(
                self.vmin,
                self.vmax,
                num_bins,
                device=self.device,
            )
        else:
            self.bin_size = 0.0
            self.bin_centers = None

    # ------------------------------------------------------------------
    # 设备迁移
    # ------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str,
    ) -> "TwoHotProcessor":
        """
        将 ``bin_centers`` 迁移到指定设备并返回自身。

        参数:
            device: 目标设备。

        返回:
            就地修改后的自身，便于链式调用。
        """
        self.device = torch.device(device)
        if self.bin_centers is not None:
            self.bin_centers = self.bin_centers.to(self.device)
        return self

    # ------------------------------------------------------------------
    # 编码 / 解码
    # ------------------------------------------------------------------

    def encode(
        self,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """
        把标量目标编码为 two-hot 向量。

        参数:
            scalars: 形状 ``(..., 1)`` 的标量张量。

        返回:
            * ``num_bins == 0``: 直接返回 ``scalars``；
            * ``num_bins == 1``: 返回 ``symlog(scalars)``，形状不变；
            * ``num_bins >= 2``: 返回形状 ``(..., num_bins)`` 的
              two-hot 概率向量。
        """
        if self.num_bins == 0:
            return scalars
        if self.num_bins == 1:
            return symlog(scalars)

        compressed = symlog(scalars)
        clamped = torch.clamp(compressed, self.vmin, self.vmax).squeeze(-1)

        bin_index = torch.floor((clamped - self.vmin) / self.bin_size).long()
        bin_index = torch.clamp(bin_index, 0, self.num_bins - 1)
        bin_offset = (
            (clamped - self.vmin) / self.bin_size - bin_index.float()
        ).unsqueeze(-1)

        next_bin_index = torch.clamp(bin_index + 1, 0, self.num_bins - 1)

        two_hot = torch.zeros(
            *clamped.shape,
            self.num_bins,
            device=scalars.device,
            dtype=scalars.dtype,
        )
        two_hot.scatter_(-1, bin_index.unsqueeze(-1), 1.0 - bin_offset)
        two_hot.scatter_add_(-1, next_bin_index.unsqueeze(-1), bin_offset)
        return two_hot

    def decode(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        把模型输出的 logits 解码为标量预测。

        参数:
            logits: 形状 ``(..., num_bins)`` 的 logits 张量；
                ``num_bins <= 1`` 时也可传形状 ``(..., 1)`` 的张量。

        返回:
            形状 ``(..., 1)`` 的标量预测。
        """
        if self.num_bins == 0:
            return logits
        if self.num_bins == 1:
            return symexp(logits)

        bin_centers = self.bin_centers.to(logits.device)
        probabilities = F.softmax(logits, dim=-1)
        weighted_sum = (probabilities * bin_centers).sum(dim=-1, keepdim=True)
        return symexp(weighted_sum)

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------

    def cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target_scalars: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 two-hot 离散回归的交叉熵损失。

        参数:
            logits: 形状 ``(..., num_bins)`` 的模型输出。
            target_scalars: 形状 ``(..., 1)`` 的标量目标。

        返回:
            形状 ``(..., 1)`` 的逐样本损失张量；调用方根据需要再做
            ``mean`` / ``sum`` 聚合。

        异常:
            ValueError: 当 ``num_bins == 0`` 时抛出（此时无 logits 概念，
                请直接使用 MSE）。
        """
        if self.num_bins == 0:
            raise ValueError(
                "num_bins=0 时不应使用 cross_entropy_loss，请改用 MSE。"
            )
        if self.num_bins == 1:
            # 退化为 symlog 标量回归
            return F.mse_loss(
                symexp(logits),
                target_scalars,
                reduction="none",
            )

        log_probabilities = F.log_softmax(logits, dim=-1)
        target_distribution = self.encode(target_scalars)
        return -(target_distribution * log_probabilities).sum(
            dim=-1,
            keepdim=True,
        )
