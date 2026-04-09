"""
``SparseMoERewardModel`` 简易冒烟测试脚本。

依次实例化奖励模型及其内部组件，打印基本信息并按顺序调用其
公开方法，便于人工检查是否有错误。

运行方式::

    python -m tests.test_reward
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import torch

from src.models.reward import (
    NoisyTopKRouter,
    SelfAttentionExpert,
    SparseMoERewardModel,
)


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
    if isinstance(value, dict):
        return {k: _summarize(v) for k, v in value.items()}
    return value


def main() -> None:
    """脚本入口：依次测试 SelfAttentionExpert / NoisyTopKRouter / SparseMoERewardModel。"""
    print("=" * 60)
    print("SparseMoERewardModel 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)

    n_agents = 2
    latent_dim = 64
    action_dim = 8
    token_dim = latent_dim + action_dim
    batch_size = 4
    n_experts = 4
    top_k = 2
    num_bins = 51

    # ------------------------------------------------------------------
    # 1. SelfAttentionExpert
    # ------------------------------------------------------------------
    print("\n[1] SelfAttentionExpert")
    expert = SelfAttentionExpert(
        token_dim=token_dim,
        n_heads=2,
        ffn_hidden_dim=128,
    )
    print(f"  类名     : {type(expert).__name__}")
    print(f"  token_dim: {token_dim}")
    _run_step(
        "SelfAttentionExpert.forward (B, N_a, D)",
        lambda: expert(torch.randn(batch_size, n_agents, token_dim)),
    )

    # ------------------------------------------------------------------
    # 2. NoisyTopKRouter
    # ------------------------------------------------------------------
    print("\n[2] NoisyTopKRouter")
    router = NoisyTopKRouter(
        in_dim=n_agents * token_dim,
        n_experts=n_experts,
        top_k=top_k,
    )
    print(f"  类名     : {type(router).__name__}")
    print(f"  in_dim   : {router.in_dim}")
    print(f"  n_experts: {router.n_experts}")
    print(f"  top_k    : {router.top_k}")

    _run_step(
        "NoisyTopKRouter.forward train 模式",
        lambda: router(torch.randn(batch_size, n_agents * token_dim)),
    )
    router.eval()
    _run_step(
        "NoisyTopKRouter.forward eval 模式",
        lambda: router(torch.randn(batch_size, n_agents * token_dim)),
    )
    router.train()
    _run_step(
        "NoisyTopKRouter top_k > n_experts (应抛 ValueError)",
        lambda: NoisyTopKRouter(
            in_dim=8,
            n_experts=2,
            top_k=4,
        ),
    )

    # ------------------------------------------------------------------
    # 3. SparseMoERewardModel
    # ------------------------------------------------------------------
    print("\n[3] SparseMoERewardModel")
    reward_model = SparseMoERewardModel(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        n_experts=n_experts,
        top_k=top_k,
        num_bins=num_bins,
        n_attention_heads=2,
        expert_ffn_hidden_dim=128,
        head_hidden_dim=64,
    )
    print(f"  类名      : {type(reward_model).__name__}")
    print(f"  n_agents  : {reward_model.n_agents}")
    print(f"  latent_dim: {reward_model.latent_dim}")
    print(f"  action_dim: {reward_model.action_dim}")
    print(f"  token_dim : {reward_model.token_dim}")
    print(f"  n_experts : {reward_model.n_experts}")
    print(f"  top_k     : {reward_model.top_k}")
    print(f"  num_bins  : {reward_model.num_bins}")
    n_params = sum(p.numel() for p in reward_model.parameters())
    print(f"  参数总数  : {n_params}")

    latents = torch.randn(batch_size, n_agents, latent_dim)
    actions = torch.randn(batch_size, n_agents, action_dim)

    _run_step(
        "forward (返回 logits 与 aux)",
        lambda: reward_model(latents, actions),
    )
    _run_step(
        "predict (仅返回 logits)",
        lambda: reward_model.predict(latents, actions),
    )

    _run_step(
        "反向传播 (logits + aux loss)",
        lambda: (
            (
                lambda outputs: (
                    outputs[0].pow(2).mean()
                    + outputs[1]["load_balancing_loss"]
                ).backward()
            )(reward_model(latents, actions))
            or "backward 完成"
        ),
    )

    reward_model.eval()
    _run_step(
        "eval 模式 forward",
        lambda: reward_model(latents, actions),
    )
    reward_model.train()

    # ------------------------------------------------------------------
    # 异常路径
    # ------------------------------------------------------------------
    _run_step(
        "forward latents 维度错误 (应抛 ValueError)",
        lambda: reward_model(torch.randn(batch_size, latent_dim), actions),
    )
    _run_step(
        "forward batch 不一致 (应抛 ValueError)",
        lambda: reward_model(
            latents,
            torch.randn(batch_size + 1, n_agents, action_dim),
        ),
    )
    _run_step(
        "forward n_agents 不一致 (应抛 ValueError)",
        lambda: reward_model(
            torch.randn(batch_size, n_agents + 1, latent_dim),
            torch.randn(batch_size, n_agents + 1, action_dim),
        ),
    )
    _run_step(
        "forward latent_dim 不一致 (应抛 ValueError)",
        lambda: reward_model(
            torch.randn(batch_size, n_agents, latent_dim + 1),
            actions,
        ),
    )
    _run_step(
        "forward action_dim 不一致 (应抛 ValueError)",
        lambda: reward_model(
            latents,
            torch.randn(batch_size, n_agents, action_dim + 1),
        ),
    )
    _run_step(
        "构造 token_dim 无法被 n_attention_heads 整除 (应抛 ValueError)",
        lambda: SparseMoERewardModel(
            n_agents=n_agents,
            latent_dim=latent_dim,
            action_dim=action_dim,
            n_attention_heads=5,
        ),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
