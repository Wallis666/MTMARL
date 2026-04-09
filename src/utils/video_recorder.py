"""
轻量级视频录制工具。

把一段 ``(T, H, W, 3)`` 的 RGB 帧序列写成 mp4 文件，便于在评估
阶段把 actor 的确定性轨迹可视化保存下来。

依赖:

* ``imageio`` 与 ``imageio-ffmpeg``: ``pip install imageio imageio-ffmpeg``

仅在被调用时才 ``import imageio``，避免对未启用渲染功能的训练
任务产生额外依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_video(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    fps: int = 30,
) -> Path:
    """
    把一组 RGB 帧写成 mp4 文件。

    参数:
        frames: 形如 ``[(H, W, 3) uint8, ...]`` 的帧序列。
        output_path: 输出文件路径，所在目录不存在时自动创建。
        fps: 写入视频的帧率。

    返回:
        写入完成后的输出路径。

    异常:
        ValueError: 当 ``frames`` 为空时抛出。
        ImportError: 当未安装 ``imageio`` 或 ``imageio-ffmpeg`` 时抛出。
    """
    if len(frames) == 0:
        raise ValueError("frames 为空，无法写入视频。")

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "录制视频需要 imageio 与 imageio-ffmpeg，请先 "
            "`pip install imageio imageio-ffmpeg`。"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()

    return output_path
