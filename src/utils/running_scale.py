"""
基于滑动百分位数的运行时尺度估计。

参考 m3w-marl ``world_models.RunningScale`` 与 TD-MPC2 的同名工具：
对一批最近的 Q 值（或 reward / advantage 等任意标量张量）计算其
``[5%, 95%]`` 分位数差作为"动态范围"，再用 EMA 平滑成一个标量
``S``，最后把目标值除以 ``S`` 做归一化。

这种"自适应除以分位数差"的归一化能在不同任务、不同 reward 量级
之间显著稳定 critic 训练，是 TD-MPC2 论文里能在 dm_control + meta-world
上"一组超参跑全部任务"的关键技巧之一。

典型用法::

    scale = RunningScale(tau=0.01, device="cuda")
    # 每次 critic 更新时:
    scaled_target = scale(td_target, update=True)
    critic_loss = ce(q_logits, scaled_target)

注意:

* 本类不持有可学习参数，但 ``value`` 与 ``percentiles`` 是 buffer
  形式的张量，需通过 :meth:`to` 同步迁移设备。
* ``num_samples == 1`` 时分位数差无意义，会退化为不更新（保持上一次
  ``S`` 不变）。
"""

from __future__ import annotations

import torch


class RunningScale:
    """
    用 ``[5%, 95%]`` 分位数差做 EMA 的运行时尺度估计器。

    内部维护一个标量 ``self._value``，每次 :meth:`update` 都把它向
    最新批次的分位数差靠拢一步：

        ``S ← (1 - τ) · S + τ · max(p95 - p5, 1)``
    """

    def __init__(
        self,
        tau: float = 0.01,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        参数:
            tau: EMA 平滑系数，越大越快跟上最新批次。
            device: 内部 buffer 张量所在设备。
            dtype: 内部 buffer 张量的数据类型。

        异常:
            ValueError: 当 ``tau`` 不在 ``(0, 1]`` 范围内时抛出。
        """
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau 必须在 (0, 1] 范围内，实际得到 {tau}。")

        self.tau = tau
        self.device = torch.device(device)
        self.dtype = dtype

        self._value = torch.ones((), device=self.device, dtype=self.dtype)
        self._percentiles = torch.tensor(
            [5.0, 95.0],
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    # 设备 / 数据迁移
    # ------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str,
    ) -> "RunningScale":
        """把内部 buffer 迁移到指定设备并返回自身。"""
        self.device = torch.device(device)
        self._value = self._value.to(self.device)
        self._percentiles = self._percentiles.to(self.device)
        return self

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    @property
    def value(self) -> float:
        """当前尺度估计的标量值。"""
        return float(self._value.item())

    def update(
        self,
        samples: torch.Tensor,
    ) -> None:
        """
        用最新一批样本更新尺度估计。

        参数:
            samples: 任意形状的浮点张量；会被 detach 并展平到一维后
                统计 ``[5%, 95%]`` 分位数。当样本数小于 2 时跳过更新。
        """
        flattened = samples.detach().to(self.device, dtype=self.dtype).flatten()
        if flattened.numel() < 2:
            return
        percentile_values = self._compute_percentiles(flattened)
        new_value = torch.clamp(
            percentile_values[1] - percentile_values[0],
            min=1.0,
        )
        self._value.lerp_(new_value, self.tau)

    def __call__(
        self,
        samples: torch.Tensor,
        update: bool = False,
    ) -> torch.Tensor:
        """
        用当前尺度归一化输入。

        参数:
            samples: 待归一化的张量。
            update: 是否在归一化前先用 ``samples`` 更新尺度。

        返回:
            归一化后的张量，与输入同形状。
        """
        if update:
            self.update(samples)
        return samples * (1.0 / self._value)

    # ------------------------------------------------------------------
    # 状态保存 / 恢复
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """返回内部 buffer 的快照，便于检查点保存。"""
        return {
            "value": self._value.clone(),
            "percentiles": self._percentiles.clone(),
        }

    def load_state_dict(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        """从 :meth:`state_dict` 的输出恢复内部 buffer。"""
        self._value.copy_(state["value"].to(self.device, dtype=self.dtype))
        self._percentiles.copy_(
            state["percentiles"].to(self.device, dtype=self.dtype),
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _compute_percentiles(
        self,
        flattened: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算一维张量的 ``[5%, 95%]`` 分位数。

        采用与 m3w 实现一致的线性插值方式，避免不同 PyTorch 版本之间
        ``torch.quantile`` 的细微差异。
        """
        sorted_values, _ = torch.sort(flattened)
        positions = self._percentiles * (sorted_values.numel() - 1) / 100.0
        floor_indices = torch.floor(positions).long()
        ceil_indices = torch.clamp(
            floor_indices + 1,
            max=sorted_values.numel() - 1,
        )
        ceil_weights = positions - floor_indices.float()
        floor_weights = 1.0 - ceil_weights

        return (
            sorted_values[floor_indices] * floor_weights
            + sorted_values[ceil_indices] * ceil_weights
        )

    def __repr__(self) -> str:
        return f"RunningScale(value={self.value:.4f}, tau={self.tau})"
