"""
``SoftMoEDynamics`` 简易冒烟测试脚本。

依次实例化模型，打印基本信息并按顺序调用其公开方法，便于人工
检查是否有错误。

运行方式::

    python -m tests.test_dynamics
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from src.models.dynamics import SoftMoEDynamics


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
    """脚本入口：依次测试 SoftMoEDynamics。"""
    print("=" * 60)
    print("SoftMoEDynamics 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)

    n_agents = 2
    latent_dim = 64
    action_dim = 6
    batch_size = 4
    horizon = 5
    n_experts = 4
    n_slots_per_expert = 2

    dynamics = SoftMoEDynamics(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        n_experts=n_experts,
        n_slots_per_expert=n_slots_per_expert,
        hidden_dims=(128, 128),
        simnorm_group_dim=8,
    )

    print("\n[1] 基本信息")
    print(f"  类名               : {type(dynamics).__name__}")
    print(f"  n_agents           : {dynamics.n_agents}")
    print(f"  latent_dim         : {dynamics.latent_dim}")
    print(f"  action_dim         : {dynamics.action_dim}")
    print(f"  token_dim          : {dynamics.token_dim}")
    print(f"  n_experts          : {dynamics.n_experts}")
    print(f"  n_slots_per_expert : {dynamics.n_slots_per_expert}")
    print(f"  routing_phi.shape  : {tuple(dynamics.routing_phi.shape)}")
    print(f"  experts[0]         :\n{dynamics.experts[0]}")
    n_params = sum(p.numel() for p in dynamics.parameters())
    print(f"  参数总数           : {n_params}")

    latents = torch.randn(batch_size, n_agents, latent_dim)
    actions = torch.randn(batch_size, n_agents, action_dim)

    _run_step(
        "forward (B, N_a, D)",
        lambda: dynamics(latents, actions),
    )

    _run_step(
        "rollout (T, B, N_a, action_dim)",
        lambda: dynamics.rollout(
            latents,
            torch.randn(horizon, batch_size, n_agents, action_dim),
        ),
    )

    _run_step(
        "反向传播 (loss.backward)",
        lambda: (
            dynamics(latents, actions).pow(2).mean().backward()
            or "backward 完成"
        ),
    )

    _run_step(
        "eval 模式 forward",
        lambda: (dynamics.eval(), dynamics(latents, actions))[1],
    )
    dynamics.train()

    # ------------------------------------------------------------------
    # 异常路径
    # ------------------------------------------------------------------
    _run_step(
        "forward latents 维度错误 (应抛 ValueError)",
        lambda: dynamics(torch.randn(batch_size, latent_dim), actions),
    )
    _run_step(
        "forward batch 不一致 (应抛 ValueError)",
        lambda: dynamics(
            latents,
            torch.randn(batch_size + 1, n_agents, action_dim),
        ),
    )
    _run_step(
        "forward n_agents 不一致 (应抛 ValueError)",
        lambda: dynamics(
            torch.randn(batch_size, n_agents + 1, latent_dim),
            torch.randn(batch_size, n_agents + 1, action_dim),
        ),
    )
    _run_step(
        "forward latent_dim 不一致 (应抛 ValueError)",
        lambda: dynamics(
            torch.randn(batch_size, n_agents, latent_dim + 1),
            actions,
        ),
    )
    _run_step(
        "forward action_dim 不一致 (应抛 ValueError)",
        lambda: dynamics(
            latents,
            torch.randn(batch_size, n_agents, action_dim + 1),
        ),
    )
    _run_step(
        "rollout action_sequence 维度错误 (应抛 ValueError)",
        lambda: dynamics.rollout(
            latents,
            torch.randn(batch_size, n_agents, action_dim),
        ),
    )
    _run_step(
        "构造 latent_dim 不可整除 (应抛 ValueError)",
        lambda: SoftMoEDynamics(
            n_agents=n_agents,
            latent_dim=10,
            action_dim=action_dim,
            simnorm_group_dim=8,
        ),
    )
    _run_step(
        "构造 n_experts=0 (应抛 ValueError)",
        lambda: SoftMoEDynamics(
            n_agents=n_agents,
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_experts=0,
        ),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
