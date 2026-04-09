"""
PyTorch 设备 / 数据类型胶水工具。

把训练循环里散落各处的 ``np.ndarray ↔ torch.Tensor`` / 设备迁移 /
模块批量 ``to(device)`` 等样板代码集中到一个文件，避免每个 trainer
/ runner 重复实现。

提供:

* :func:`to_tensor` —— 把 numpy 数组 / 标量 / 已是张量的对象转成
  指定 device 与 dtype 的 ``torch.Tensor``；
* :func:`to_numpy` —— 把张量 detach 后搬回 CPU 并转成 numpy；
* :func:`tree_to_tensor` / :func:`tree_to_numpy` —— 对字典 / 列表 /
  元组做递归转换，专门服务于 ``ReplayBuffer.sample`` 的 batch 字典；
* :func:`move_modules` —— 一次性把多个 ``nn.Module`` 迁移到目标设备。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


def to_tensor(
    value: Any,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    把任意支持的输入转成指定设备 / 数据类型的张量。

    参数:
        value: numpy 数组 / Python 标量 / 已是 ``torch.Tensor`` 的对象。
        device: 目标设备。
        dtype: 目标 dtype。

    返回:
        位于 ``device`` 上、dtype 为 ``dtype`` 的张量；输入若已经满足
        条件则直接返回（避免不必要的拷贝）。
    """
    target_device = torch.device(device)
    if isinstance(value, torch.Tensor):
        if value.device == target_device and value.dtype == dtype:
            return value
        return value.to(device=target_device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=target_device)


def to_numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    把张量 detach 后搬回 CPU 并转成 numpy 数组。

    参数:
        tensor: 任意 ``torch.Tensor``。

    返回:
        与输入同形状的 numpy 数组。
    """
    return tensor.detach().cpu().numpy()


def tree_to_tensor(
    tree: Any,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Any:
    """
    对字典 / 列表 / 元组结构做递归 :func:`to_tensor` 转换。

    参数:
        tree: 含 numpy 数组的嵌套结构。
        device: 目标设备。
        dtype: 目标 dtype。

    返回:
        与输入同结构、所有叶子均为张量的对象；非数组叶子原样保留。
    """
    if isinstance(tree, dict):
        return {key: tree_to_tensor(value, device, dtype) for key, value in tree.items()}
    if isinstance(tree, list):
        return [tree_to_tensor(item, device, dtype) for item in tree]
    if isinstance(tree, tuple):
        return tuple(tree_to_tensor(item, device, dtype) for item in tree)
    if isinstance(tree, (np.ndarray, torch.Tensor)) or np.isscalar(tree):
        return to_tensor(tree, device=device, dtype=dtype)
    return tree


def tree_to_numpy(
    tree: Any,
) -> Any:
    """对字典 / 列表 / 元组结构做递归 :func:`to_numpy` 转换。"""
    if isinstance(tree, dict):
        return {key: tree_to_numpy(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [tree_to_numpy(item) for item in tree]
    if isinstance(tree, tuple):
        return tuple(tree_to_numpy(item) for item in tree)
    if isinstance(tree, torch.Tensor):
        return to_numpy(tree)
    return tree


def move_modules(
    modules: Iterable[nn.Module],
    device: torch.device | str,
) -> None:
    """
    一次性把多个 ``nn.Module`` 就地迁移到目标设备。

    参数:
        modules: 任意可迭代的 ``nn.Module`` 集合。
        device: 目标设备。
    """
    target_device = torch.device(device)
    for module in modules:
        module.to(target_device)
