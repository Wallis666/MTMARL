"""
轻量级训练日志记录器。

提供一个最小可用的 :class:`Logger`，同时把训练指标写到:

* **stdout**：人类可读的单行 ``key=value`` 摘要，便于跟踪进度；
* **CSV 文件**：结构化记录，便于后续画图与对照实验。

设计原则:

* **零外部依赖**：不绑定 wandb / tensorboard，所有持久化都用 Python
  标准库的 ``csv``，跑通流水线后想接更强的 logger 只需替换 :meth:`log`。
* **延迟列扩展**：第一次写入时尚未见过的 key 会自动加入 CSV 表头；
  后续若出现新 key 则在内存里重排表头并整文件改写一次（小规模训练
  完全够用）。
* **append-friendly**：支持续训：构造时若文件已存在则读取已有表头并
  追加，避免覆盖历史。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


class Logger:
    """
    最小可用的 stdout + CSV 训练日志记录器。

    用法::

        logger = Logger(log_dir="runs/exp1")
        logger.log({"loss": 0.1, "ep_return": 12.3}, step=1000)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str | Path,
        csv_filename: str = "metrics.csv",
        stdout: bool = True,
    ) -> None:
        """
        参数:
            log_dir: 日志输出目录，不存在则自动创建。
            csv_filename: CSV 文件名。
            stdout: 是否同时把每条记录打印到 stdout。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / csv_filename
        self.stdout = stdout
        self._start_time = time.time()

        self._field_names: list[str] = ["step", "wall_time"]
        self._rows: list[dict[str, Any]] = []

        if self.csv_path.exists():
            self._load_existing()

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """
        写入一条训练记录。

        参数:
            metrics: ``{key: value}`` 字典，值需可被 ``str`` 化。
            step: 当前训练步数，会作为 CSV 第一列。
        """
        row: dict[str, Any] = {
            "step": step,
            "wall_time": round(time.time() - self._start_time, 3),
        }
        row.update(metrics)

        new_keys = [key for key in row.keys() if key not in self._field_names]
        if new_keys:
            self._field_names.extend(new_keys)
            self._rewrite_csv_with_new_header(extra_row=row)
        else:
            self._append_csv_row(row)

        self._rows.append(row)

        if self.stdout:
            self._print_row(row)

    def close(self) -> None:
        """
        关闭 logger（当前实现仅作为占位，便于未来接入需要显式关闭的后端）。
        """
        return

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        """读取既有 CSV 的表头与全部行，便于续训追加。"""
        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is not None:
                self._field_names = list(reader.fieldnames)
            for row in reader:
                self._rows.append(dict(row))

    def _append_csv_row(
        self,
        row: dict[str, Any],
    ) -> None:
        """以追加模式写入一行；表头不存在时先写表头。"""
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._field_names)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self._field_names})

    def _rewrite_csv_with_new_header(
        self,
        extra_row: dict[str, Any],
    ) -> None:
        """
        当出现新字段时，重写整个 CSV 文件以包含完整表头。

        参数:
            extra_row: 触发重写的最新一行，会和历史行一起写回。
        """
        with self.csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._field_names)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(
                    {key: row.get(key, "") for key in self._field_names},
                )
            writer.writerow(
                {key: extra_row.get(key, "") for key in self._field_names},
            )

    def _print_row(
        self,
        row: dict[str, Any],
    ) -> None:
        """把一行记录格式化为单行 ``key=value`` 字符串并打印。"""
        ordered = [
            f"{key}={self._format_value(row.get(key, ''))}"
            for key in self._field_names
        ]
        print("[log] " + "  ".join(ordered))

    @staticmethod
    def _format_value(
        value: Any,
    ) -> str:
        """对浮点值使用 4 位小数，其余直接 ``str`` 化。"""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
