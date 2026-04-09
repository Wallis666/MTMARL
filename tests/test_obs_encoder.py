"""
``ObsEncoder`` 简易冒烟测试脚本。

依次实例化 ``SimNorm`` / ``NormedLinear`` / ``ObsEncoder``，打印基本
信息并按顺序调用其公开方法，便于人工检查是否有错误。

运行方式::

    python -m tests.test_obs_encoder
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from src.models.obs_encoder import NormedLinear, ObsEncoder, SimNorm


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
        return f"tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    return value


def main() -> None:
    """脚本入口：依次测试 SimNorm / NormedLinear / ObsEncoder。"""
    print("=" * 60)
    print("ObsEncoder 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # 1. SimNorm
    # ------------------------------------------------------------------
    print("\n[1] SimNorm")
    simnorm = SimNorm(group_dim=4)
    print(f"  类名     : {type(simnorm).__name__}")
    print(f"  group_dim: {simnorm.group_dim}")
    print(f"  extra_repr: {simnorm.extra_repr()}")
    _run_step(
        "SimNorm.forward (B=2, D=16)",
        lambda: simnorm(torch.randn(2, 16)),
    )
    _run_step(
        "SimNorm 输出按组求和应≈1",
        lambda: simnorm(torch.randn(2, 16)).view(2, -1, 4).sum(dim=-1),
    )

    # ------------------------------------------------------------------
    # 2. NormedLinear
    # ------------------------------------------------------------------
    print("\n[2] NormedLinear")
    normed = NormedLinear(in_features=8, out_features=16)
    print(f"  类名: {type(normed).__name__}")
    print(f"  in -> out: {normed.in_features} -> {normed.out_features}")
    print(f"  activation: {type(normed.activation).__name__}")
    _run_step(
        "NormedLinear.forward (B=4, D=8)",
        lambda: normed(torch.randn(4, 8)),
    )

    # ------------------------------------------------------------------
    # 3. ObsEncoder
    # ------------------------------------------------------------------
    print("\n[3] ObsEncoder")
    obs_dim = 64
    latent_dim = 128
    encoder = ObsEncoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=(64, 64),
        simnorm_group_dim=8,
    )
    print(f"  类名      : {type(encoder).__name__}")
    print(f"  obs_dim   : {encoder.obs_dim}")
    print(f"  latent_dim: {encoder.latent_dim}")
    print(f"  trunk     :\n{encoder.trunk}")
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  参数总数  : {n_params}")

    _run_step(
        "ObsEncoder.forward (B=4, obs_dim)",
        lambda: encoder(torch.randn(4, obs_dim)),
    )
    _run_step(
        "ObsEncoder.forward (num_envs=2, n_agents=3, obs_dim)",
        lambda: encoder(torch.randn(2, 3, obs_dim)),
    )
    _run_step(
        "ObsEncoder.forward (T=5, B=4, n_agents=3, obs_dim)",
        lambda: encoder(torch.randn(5, 4, 3, obs_dim)),
    )
    _run_step(
        "ObsEncoder.forward 末维不匹配 (应抛出 ValueError)",
        lambda: encoder(torch.randn(4, obs_dim + 1)),
    )
    _run_step(
        "ObsEncoder 反向传播 (loss.backward)",
        lambda: (
            encoder(torch.randn(4, obs_dim)).pow(2).mean().backward()
            or "backward 完成"
        ),
    )

    # ------------------------------------------------------------------
    # 4. 异常构造
    # ------------------------------------------------------------------
    _run_step(
        "ObsEncoder 非法 latent_dim (应抛出 ValueError)",
        lambda: ObsEncoder(obs_dim=16, latent_dim=10, simnorm_group_dim=8),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
