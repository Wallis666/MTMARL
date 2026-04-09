"""
``ReplayBuffer`` 简易冒烟测试脚本。

依次实例化缓冲区，模拟多环境写入并按顺序调用其公开方法，便于
人工检查是否有错误。

运行方式::

    python -m tests.test_replay_buffer
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import numpy as np

from src.buffers.replay_buffer import ReplayBuffer


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
    """对返回值做精简表示，避免在终端打印整张大数组。"""
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return {k: _summarize(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_summarize(item) for item in value)
    return value


def _make_fake_transition(
    num_envs: int,
    n_agents: int,
    obs_dim: int,
    shared_state_dim: int,
    action_dim: int,
    step: int,
) -> dict[str, np.ndarray]:
    """构造一组随机 transition，``step`` 用于控制 done 触发频率。"""
    rng = np.random.default_rng(step)
    dones = np.zeros(num_envs, dtype=np.bool_)
    if step > 0 and step % 5 == 0:
        dones[0] = True  # 周期性触发 episode 结束，便于测试 _end_flag
    return {
        "obs": rng.standard_normal(
            (num_envs, n_agents, obs_dim),
        ).astype(np.float32),
        "shared_obs": rng.standard_normal(
            (num_envs, n_agents, shared_state_dim),
        ).astype(np.float32),
        "actions": rng.standard_normal(
            (num_envs, n_agents, action_dim),
        ).astype(np.float32),
        "rewards": rng.standard_normal(num_envs).astype(np.float32),
        "dones": dones,
        "terminations": dones.copy(),
        "next_obs": rng.standard_normal(
            (num_envs, n_agents, obs_dim),
        ).astype(np.float32),
        "next_shared_obs": rng.standard_normal(
            (num_envs, n_agents, shared_state_dim),
        ).astype(np.float32),
    }


def main() -> None:
    """脚本入口：依次测试 ReplayBuffer。"""
    print("=" * 60)
    print("ReplayBuffer 冒烟测试")
    print("=" * 60)

    np.random.seed(0)

    num_envs = 4
    n_agents = 2
    obs_dim = 16
    shared_state_dim = 24
    action_dim = 6
    buffer_size = 64
    n_step = 3
    gamma = 0.99

    buffer = ReplayBuffer(
        buffer_size=buffer_size,
        num_envs=num_envs,
        n_agents=n_agents,
        obs_dim=obs_dim,
        shared_state_dim=shared_state_dim,
        action_dim=action_dim,
        n_step=n_step,
        gamma=gamma,
    )

    print("\n[1] 基本信息")
    print(f"  类名             : {type(buffer).__name__}")
    print(f"  buffer_size      : {buffer.buffer_size}")
    print(f"  num_envs         : {buffer.num_envs}")
    print(f"  n_agents         : {buffer.n_agents}")
    print(f"  obs_dim          : {buffer.obs_dim}")
    print(f"  shared_state_dim : {buffer.shared_state_dim}")
    print(f"  action_dim       : {buffer.action_dim}")
    print(f"  n_step           : {buffer.n_step}")
    print(f"  gamma            : {buffer.gamma}")
    print(f"  initial len      : {len(buffer)}")

    # ------------------------------------------------------------------
    # 写入 (写到刚好填满后再多写几步触发环形覆盖)
    # ------------------------------------------------------------------
    n_insert_steps = (buffer_size // num_envs) + 4
    _run_step(
        f"insert {n_insert_steps} 步 (含环形覆盖)",
        lambda: [
            buffer.insert(
                **_make_fake_transition(
                    num_envs,
                    n_agents,
                    obs_dim,
                    shared_state_dim,
                    action_dim,
                    step,
                ),
            )
            for step in range(n_insert_steps)
        ]
        and f"len={len(buffer)}, write_index={buffer.write_index}",
    )

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------
    _run_step(
        "sample(batch_size=8)",
        lambda: buffer.sample(batch_size=8),
    )

    _run_step(
        "sample(batch_size=8) 字段形状检查",
        lambda: {k: v.shape for k, v in buffer.sample(batch_size=8).items()},
    )

    # ------------------------------------------------------------------
    # sample_horizon
    # ------------------------------------------------------------------
    _run_step(
        "sample_horizon(batch_size=4, horizon=5)",
        lambda: buffer.sample_horizon(batch_size=4, horizon=5),
    )

    _run_step(
        "sample_horizon 字段形状检查",
        lambda: {
            k: v.shape
            for k, v in buffer.sample_horizon(batch_size=4, horizon=5).items()
        },
    )

    # ------------------------------------------------------------------
    # 杂项
    # ------------------------------------------------------------------
    _run_step(
        "get_mean_reward()",
        lambda: buffer.get_mean_reward(),
    )
    _run_step(
        "len(buffer)",
        lambda: len(buffer),
    )

    # ------------------------------------------------------------------
    # 异常路径
    # ------------------------------------------------------------------
    _run_step(
        "insert num_envs 不一致 (应抛 ValueError)",
        lambda: buffer.insert(
            obs=np.zeros((num_envs + 1, n_agents, obs_dim), dtype=np.float32),
            shared_obs=np.zeros(
                (num_envs + 1, n_agents, shared_state_dim),
                dtype=np.float32,
            ),
            actions=np.zeros(
                (num_envs + 1, n_agents, action_dim),
                dtype=np.float32,
            ),
            rewards=np.zeros(num_envs + 1, dtype=np.float32),
            dones=np.zeros(num_envs + 1, dtype=np.bool_),
            terminations=np.zeros(num_envs + 1, dtype=np.bool_),
            next_obs=np.zeros(
                (num_envs + 1, n_agents, obs_dim),
                dtype=np.float32,
            ),
            next_shared_obs=np.zeros(
                (num_envs + 1, n_agents, shared_state_dim),
                dtype=np.float32,
            ),
        ),
    )
    _run_step(
        "sample 超过容量 (应抛 ValueError)",
        lambda: buffer.sample(batch_size=buffer_size + 10),
    )
    _run_step(
        "sample_horizon horizon=0 (应抛 ValueError)",
        lambda: buffer.sample_horizon(batch_size=4, horizon=0),
    )
    _run_step(
        "构造 buffer_size 非 num_envs 整数倍 (应抛 ValueError)",
        lambda: ReplayBuffer(
            buffer_size=63,
            num_envs=4,
            n_agents=n_agents,
            obs_dim=obs_dim,
            shared_state_dim=shared_state_dim,
            action_dim=action_dim,
        ),
    )
    _run_step(
        "构造 n_step=0 (应抛 ValueError)",
        lambda: ReplayBuffer(
            buffer_size=64,
            num_envs=4,
            n_agents=n_agents,
            obs_dim=obs_dim,
            shared_state_dim=shared_state_dim,
            action_dim=action_dim,
            n_step=0,
        ),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
