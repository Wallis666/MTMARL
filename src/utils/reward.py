"""
奖励工具模块。

提供软指示函数，用于评估数值是否落在指定区间内。
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


# tolerance() 在距离 bounds 区间 margin 处的默认返回值
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(
    x: NDArray,
    value_at_1: float,
    sigmoid: str,
) -> NDArray:
    """
    当 x == 0 时返回 1，其余情况返回 0 到 1 之间的值。

    参数:
        x: 标量或 numpy 数组。
        value_at_1: 0 到 1 之间的浮点数，指定 x == 1 时的输出值。
        sigmoid: sigmoid 类型字符串。

    返回:
        值域为 [0.0, 1.0] 的 numpy 数组。

    异常:
        ValueError: 当 value_at_1 不在 (0, 1) 范围内时抛出；
            对于 linear、cosine、quadratic 类型允许 value_at_1 == 0。
        ValueError: 当 sigmoid 类型未知时抛出。
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` 必须为非负且小于 1，"
                f"当前值为 {value_at_1}"
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` 必须严格介于 0 和 1 之间，"
                f"当前值为 {value_at_1}"
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="invalid value encountered in cos",
            )
            cos_pi_scaled_x = np.cos(np.pi * scaled_x)
        return np.where(
            abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0
        )

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f"未知的 sigmoid 类型: {sigmoid!r}")


def tolerance(
    x: ArrayLike,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> float | NDArray:
    """
    当 x 落在 bounds 区间内时返回 1，否则返回 0 到 1 之间的值。

    参数:
        x: 标量或 numpy 数组。
        bounds: 由两个浮点数组成的元组，指定目标区间的闭区间
            (lower, upper)。可以为无穷大表示单侧无界，
            也可以两端相等表示精确目标值。
        margin: 控制输出在 x 超出边界时衰减陡度的浮点数。
            若 margin == 0，则所有超出 bounds 的 x 输出为 0；
            若 margin > 0，则输出随距离最近边界的距离以
            sigmoid 方式递减。
        sigmoid: sigmoid 类型字符串。可选值包括: gaussian、
            linear、hyperbolic、long_tail、cosine、
            tanh_squared。
        value_at_margin: 0 到 1 之间的浮点数，指定当 x 到
            最近边界的距离等于 margin 时的输出值。当
            margin == 0 时忽略此参数。

    返回:
        值域为 [0.0, 1.0] 的浮点数或 numpy 数组。

    异常:
        ValueError: 当 bounds[0] > bounds[1] 时抛出。
        ValueError: 当 margin 为负数时抛出。
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("下界必须小于等于上界")
    if margin < 0:
        raise ValueError("`margin` 必须为非负数")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(
            in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid)
        )

    return float(value) if np.isscalar(x) else value
