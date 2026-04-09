"""
``DistributionalQNetwork`` 与 ``TwinQCritic`` 简易冒烟测试脚本。

依次实例化两个组件，打印基本信息并按顺序调用其公开方法，便于
人工检查是否有错误。

运行方式::

    python -m tests.test_critic
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from src.models.critic import DistributionalQNetwork, TwinQCritic


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
    """脚本入口：依次测试 DistributionalQNetwork 与 TwinQCritic。"""
    print("=" * 60)
    print("Critic 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)

    n_agents = 2
    latent_dim = 64
    action_dim = 6
    batch_size = 4
    num_bins = 51

    latents = torch.randn(batch_size, n_agents, latent_dim)
    actions = torch.randn(batch_size, n_agents, action_dim)

    # ------------------------------------------------------------------
    # 1. DistributionalQNetwork
    # ------------------------------------------------------------------
    print("\n[1] DistributionalQNetwork")
    q_net = DistributionalQNetwork(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=(128, 128),
        num_bins=num_bins,
    )
    print(f"  类名      : {type(q_net).__name__}")
    print(f"  n_agents  : {q_net.n_agents}")
    print(f"  latent_dim: {q_net.latent_dim}")
    print(f"  action_dim: {q_net.action_dim}")
    print(f"  num_bins  : {q_net.num_bins}")
    print(f"  input_dim : {q_net.input_dim}")
    print(f"  trunk     :\n{q_net.trunk}")
    n_params = sum(p.numel() for p in q_net.parameters())
    print(f"  参数总数  : {n_params}")

    _run_step(
        "DistributionalQNetwork.forward",
        lambda: q_net(latents, actions),
    )
    _run_step(
        "末层零初始化检查 (输出应≈0)",
        lambda: q_net(latents, actions).abs().max().item(),
    )

    # ------------------------------------------------------------------
    # 2. TwinQCritic
    # ------------------------------------------------------------------
    print("\n[2] TwinQCritic")
    critic = TwinQCritic(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=(128, 128),
        num_bins=num_bins,
    )
    print(f"  类名      : {type(critic).__name__}")
    print(f"  n_agents  : {critic.n_agents}")
    print(f"  latent_dim: {critic.latent_dim}")
    print(f"  action_dim: {critic.action_dim}")
    print(f"  num_bins  : {critic.num_bins}")
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"  参数总数  : {n_params}")

    _run_step(
        "TwinQCritic.forward (返回 q1, q2)",
        lambda: critic(latents, actions),
    )
    _run_step(
        "TwinQCritic.q_min",
        lambda: critic.q_min(latents, actions),
    )

    _run_step(
        "反向传播 (q1+q2 的均方)",
        lambda: (
            (
                lambda outputs: (
                    outputs[0].pow(2).mean() + outputs[1].pow(2).mean()
                ).backward()
            )(critic(latents, actions))
            or "backward 完成"
        ),
    )

    critic.eval()
    _run_step(
        "eval 模式 forward",
        lambda: critic(latents, actions),
    )
    critic.train()

    # ------------------------------------------------------------------
    # 异常路径
    # ------------------------------------------------------------------
    _run_step(
        "forward latents 维度错误 (应抛 ValueError)",
        lambda: critic(torch.randn(batch_size, latent_dim), actions),
    )
    _run_step(
        "forward batch 不一致 (应抛 ValueError)",
        lambda: critic(
            latents,
            torch.randn(batch_size + 1, n_agents, action_dim),
        ),
    )
    _run_step(
        "forward n_agents 不一致 (应抛 ValueError)",
        lambda: critic(
            torch.randn(batch_size, n_agents + 1, latent_dim),
            torch.randn(batch_size, n_agents + 1, action_dim),
        ),
    )
    _run_step(
        "forward latent_dim 不一致 (应抛 ValueError)",
        lambda: critic(
            torch.randn(batch_size, n_agents, latent_dim + 1),
            actions,
        ),
    )
    _run_step(
        "forward action_dim 不一致 (应抛 ValueError)",
        lambda: critic(
            latents,
            torch.randn(batch_size, n_agents, action_dim + 1),
        ),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
