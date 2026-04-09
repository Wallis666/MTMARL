"""
TensorBoard 训练日志记录器。

对 ``torch.utils.tensorboard.SummaryWriter`` 做一层薄封装，让上层
代码以与 :class:`src.utils.logger.Logger` 一致的接口
``log(metrics, step)`` 写入指标，避免在 runner 内部直接耦合
TensorBoard。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    最小可用的 TensorBoard 日志记录器。

    用法::

        logger = TensorBoardLogger(log_dir="runs/exp1")
        logger.log({"loss": 0.1, "ep_return": 12.3}, step=1000)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        flush_secs: int = 10,
    ) -> None:
        """
        参数:
            log_dir: TensorBoard 事件文件输出目录，不存在则自动创建。
            flush_secs: ``SummaryWriter`` 自动 flush 的间隔秒数。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
        )

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """
        把一组标量指标写入 TensorBoard。

        参数:
            metrics: ``{key: value}`` 字典，值需可被强转为 ``float``。
                非数值或 ``NaN`` 会被静默跳过，避免污染 scalar 视图。
            step: 当前训练步数。
        """
        for key, value in metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if scalar != scalar:  # NaN 检测
                continue
            self.writer.add_scalar(key, scalar, global_step=step)

    def close(self) -> None:
        """关闭底层 ``SummaryWriter``。"""
        self.writer.flush()
        self.writer.close()
