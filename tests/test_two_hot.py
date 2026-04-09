"""
``TwoHotProcessor`` 简易冒烟测试脚本。

依次实例化三种工作模式，打印基本信息并按顺序调用其公开方法，
便于人工检查是否有错误。

运行方式::

    python -m tests.test_two_hot
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from src.utils.two_hot import TwoHotProcessor, symexp, symlog


def _run_step(
    title: str,
    func: Callable[[], Any],
) -> None:
    """执行单个测试步骤并打印结果。"""
    print(f"\n[步骤] {title}")
    try:
        result = func()
    except Exception as exc:  # noqa: BLE001
        print(f"  [失败] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    print(f"  [成功] 返回: {_summarize(result)}")


def _summarize(
    value: Any,
) -> Any:
    """对返回值做精简表示，避免在终端打印整张大张量。"""
    if isinstance(value, torch.Tensor):
        if value.numel() <= 8:
            return f"tensor({value.tolist()})"
        return f"tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    return value


def main() -> None:
    """脚本入口：依次测试 symlog/symexp 与三种 TwoHotProcessor 模式。"""
    print("=" * 60)
    print("TwoHotProcessor 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # 1. symlog / symexp
    # ------------------------------------------------------------------
    print("\n[1] symlog / symexp")
    sample = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    _run_step("symlog(sample)", lambda: symlog(sample))
    _run_step("symexp(symlog(sample))", lambda: symexp(symlog(sample)))
    _run_step(
        "可逆性检查 (max abs error)",
        lambda: (symexp(symlog(sample)) - sample).abs().max().item(),
    )

    # ------------------------------------------------------------------
    # 2. 标准 two-hot (num_bins=51)
    # ------------------------------------------------------------------
    print("\n[2] TwoHotProcessor (num_bins=51)")
    processor = TwoHotProcessor(
        num_bins=51,
        vmin=-10.0,
        vmax=10.0,
    )
    print(f"  num_bins   : {processor.num_bins}")
    print(f"  vmin / vmax: {processor.vmin} / {processor.vmax}")
    print(f"  bin_size   : {processor.bin_size}")
    print(f"  bin_centers shape: {tuple(processor.bin_centers.shape)}")

    targets = torch.tensor([[-3.7], [0.0], [2.4], [50.0]])
    _run_step("encode (B, 1) → two-hot", lambda: processor.encode(targets))
    _run_step(
        "encode 行和应≈1",
        lambda: processor.encode(targets).sum(dim=-1),
    )

    logits = torch.randn(4, 51)
    _run_step("decode (B, num_bins) → 标量", lambda: processor.decode(logits))

    _run_step(
        "round-trip: decode(logits=encode(targets).log())",
        lambda: processor.decode(
            torch.log(processor.encode(targets) + 1e-9),
        ),
    )

    _run_step(
        "cross_entropy_loss",
        lambda: processor.cross_entropy_loss(logits, targets),
    )

    # ------------------------------------------------------------------
    # 多前缀维度
    # ------------------------------------------------------------------
    multi_targets = torch.randn(2, 3, 1)
    multi_logits = torch.randn(2, 3, 51)
    _run_step(
        "encode 多前缀维度 (2, 3, 1)",
        lambda: processor.encode(multi_targets).shape,
    )
    _run_step(
        "decode 多前缀维度 (2, 3, 51)",
        lambda: processor.decode(multi_logits).shape,
    )
    _run_step(
        "cross_entropy_loss 多前缀维度",
        lambda: processor.cross_entropy_loss(multi_logits, multi_targets).shape,
    )

    # ------------------------------------------------------------------
    # 3. num_bins=1 退化模式 (symlog 标量回归)
    # ------------------------------------------------------------------
    print("\n[3] TwoHotProcessor (num_bins=1, 退化为 symlog)")
    processor1 = TwoHotProcessor(num_bins=1, vmin=-10.0, vmax=10.0)
    _run_step("encode", lambda: processor1.encode(targets))
    _run_step("decode (取 symexp)", lambda: processor1.decode(symlog(targets)))
    _run_step(
        "cross_entropy_loss (退化为 MSE)",
        lambda: processor1.cross_entropy_loss(symlog(targets), targets),
    )

    # ------------------------------------------------------------------
    # 4. num_bins=0 透传
    # ------------------------------------------------------------------
    print("\n[4] TwoHotProcessor (num_bins=0, 透传)")
    processor0 = TwoHotProcessor(num_bins=0, vmin=-10.0, vmax=10.0)
    _run_step("encode 透传", lambda: processor0.encode(targets))
    _run_step("decode 透传", lambda: processor0.decode(targets))
    _run_step(
        "cross_entropy_loss (应抛 ValueError)",
        lambda: processor0.cross_entropy_loss(targets, targets),
    )

    # ------------------------------------------------------------------
    # 5. to(device)
    # ------------------------------------------------------------------
    print("\n[5] to(device)")
    _run_step(
        "to('cpu') 链式调用",
        lambda: processor.to("cpu").bin_centers.device,
    )

    # ------------------------------------------------------------------
    # 6. 异常路径
    # ------------------------------------------------------------------
    _run_step(
        "构造 num_bins=-1 (应抛 ValueError)",
        lambda: TwoHotProcessor(num_bins=-1, vmin=-1.0, vmax=1.0),
    )
    _run_step(
        "构造 vmin >= vmax (应抛 ValueError)",
        lambda: TwoHotProcessor(num_bins=10, vmin=1.0, vmax=1.0),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
