"""
``SquashedGaussianActor`` 简易冒烟测试脚本。

依次实例化策略网络，打印基本信息并按顺序调用其公开方法，便于
人工检查是否有错误。

运行方式::

    python -m tests.test_actor
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from src.models.actor import SquashedGaussianActor


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
    if isinstance(value, tuple):
        return tuple(_summarize(item) for item in value)
    return value


def main() -> None:
    """脚本入口：依次测试 SquashedGaussianActor。"""
    print("=" * 60)
    print("SquashedGaussianActor 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)

    latent_dim = 64
    action_dim = 6
    n_agents = 2
    num_envs = 4
    action_limit = 1.0

    actor = SquashedGaussianActor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=(128, 128),
        action_limit=action_limit,
    )

    print("\n[1] 基本信息")
    print(f"  类名        : {type(actor).__name__}")
    print(f"  latent_dim  : {actor.latent_dim}")
    print(f"  action_dim  : {actor.action_dim}")
    print(f"  action_limit: {actor.action_limit}")
    print(f"  log_std_min : {actor.log_std_min.item()}")
    print(f"  log_std_max : {actor.log_std_max.item()}")
    print(f"  trunk       :\n{actor.trunk}")
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"  参数总数    : {n_params}")

    latents_2d = torch.randn(num_envs, latent_dim)
    latents_3d = torch.randn(num_envs, n_agents, latent_dim)

    _run_step(
        "forward 2D 随机采样 (B, latent_dim)",
        lambda: actor(latents_2d, stochastic=True, with_logprob=False),
    )
    _run_step(
        "forward 2D 含 logprob",
        lambda: actor(latents_2d, stochastic=True, with_logprob=True),
    )
    _run_step(
        "forward 2D 确定性 (stochastic=False)",
        lambda: actor(latents_2d, stochastic=False, with_logprob=False),
    )

    _run_step(
        "forward 3D (num_envs, n_agents, latent_dim)",
        lambda: actor(latents_3d, stochastic=True, with_logprob=True),
    )

    _run_step(
        "deterministic (..., latent_dim)",
        lambda: actor.deterministic(latents_3d),
    )

    _run_step(
        "动作范围检查 (应在 [-action_limit, +action_limit])",
        lambda: (
            lambda actions: (
                actions.abs().max().item(),
                "ok" if actions.abs().max().item() <= action_limit + 1e-5 else "越界",
            )
        )(actor.deterministic(latents_3d)),
    )

    _run_step(
        "反向传播 (含 logprob 的随机策略)",
        lambda: (
            (
                lambda outputs: (
                    outputs[0].pow(2).mean()
                    - outputs[1].mean()
                ).backward()
            )(actor(latents_3d, stochastic=True, with_logprob=True))
            or "backward 完成"
        ),
    )

    actor.eval()
    _run_step(
        "eval 模式 forward",
        lambda: actor(latents_3d, stochastic=False, with_logprob=False),
    )
    actor.train()

    # ------------------------------------------------------------------
    # 异常路径
    # ------------------------------------------------------------------
    _run_step(
        "forward latent_dim 不一致 (应抛 ValueError)",
        lambda: actor(torch.randn(num_envs, latent_dim + 1)),
    )
    _run_step(
        "构造 hidden_dims 为空 (应抛 ValueError)",
        lambda: SquashedGaussianActor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=(),
        ),
    )
    _run_step(
        "构造 log_std_min >= log_std_max (应抛 ValueError)",
        lambda: SquashedGaussianActor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            log_std_min=2.0,
            log_std_max=2.0,
        ),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
