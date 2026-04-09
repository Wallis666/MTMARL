"""
奖励函数工具模块。

提供用于强化学习奖励整形的通用工具函数，包括多种 sigmoid 形状函数
以及基于区间的容差函数。
"""

import warnings
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

# tolerance() 在距离 `bounds` 区间 `margin` 处返回的默认数值
_DEFAULT_VALUE_AT_MARGIN = 0.1

# 数值类型别名：标量或 numpy 数组
NumericArray = Union[float, np.ndarray]


def _sigmoids(
    x: ArrayLike,
    value_at_1: float,
    sigmoid: str,
) -> np.ndarray:
    """
    根据指定的 sigmoid 类型计算形状函数的值。

    当 `x` 等于 0 时返回 1，其余情况返回 0 到 1 之间的值。

    参数:
        x: 标量或 numpy 数组形式的输入。
        value_at_1: 介于 0 与 1 之间的浮点数，指定 `x == 1` 时的输出值。
        sigmoid: 字符串，sigmoid 类型。可选值为 'gaussian'、'hyperbolic'、
            'long_tail'、'reciprocal'、'cosine'、'linear'、'quadratic'、
            'tanh_squared'。

    返回:
        取值范围在 0.0 到 1.0 之间的 numpy 数组。

    异常:
        ValueError: 当 `value_at_1` 不满足取值范围要求时抛出。
            对于 'linear'、'cosine'、'quadratic' 允许 `value_at_1 == 0`，
            其余 sigmoid 必须严格满足 0 < `value_at_1` < 1。
        ValueError: 当 `sigmoid` 为未知类型时抛出。
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f'`value_at_1` 必须为非负数且小于 1，当前值为 {value_at_1}。'
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f'`value_at_1` 必须严格介于 0 和 1 之间，当前值为 {value_at_1}。'
            )

    if sigmoid == 'gaussian':
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='invalid value encountered in cos',
            )
            cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f'未知的 sigmoid 类型 {sigmoid!r}。')


def tolerance(
    x: ArrayLike,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = 'gaussian',
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> NumericArray:
    """
    计算输入相对于目标区间的容差奖励值。

    当 `x` 落在 `bounds` 指定的区间内时返回 1；当 `x` 在区间外时，
    根据所选 sigmoid 形状函数返回 0 到 1 之间的数值。

    参数:
        x: 标量或 numpy 数组形式的输入。
        bounds: 浮点数二元组，指定目标区间的闭区间 `(lower, upper)`。
            可以为无穷大表示单侧或双侧无界，也可两端相等表示精确目标值。
        margin: 浮点数，控制 `x` 偏离区间时输出衰减的陡峭程度。
            * 当 `margin == 0` 时，区间外的输出恒为 0。
            * 当 `margin > 0` 时，输出会随着到最近边界距离的增加按 sigmoid
              形状衰减。
        sigmoid: 字符串，sigmoid 类型。可选值参见 `_sigmoids` 文档。
        value_at_margin: 介于 0 与 1 之间的浮点数，指定 `x` 到最近边界
            的距离等于 `margin` 时的输出值。当 `margin == 0` 时被忽略。

    返回:
        取值范围在 0.0 到 1.0 之间的浮点数或 numpy 数组。

    异常:
        ValueError: 当 `bounds[0] > bounds[1]` 时抛出。
        ValueError: 当 `margin` 为负数时抛出。
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError('区间下界必须小于等于上界。')
    if margin < 0:
        raise ValueError('`margin` 必须为非负数。')

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(
            in_bounds,
            1.0,
            _sigmoids(d, value_at_margin, sigmoid),
        )

    return float(value) if np.isscalar(x) else value
