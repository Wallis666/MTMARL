"""
``ShareDummyVecEnv`` 与 ``ShareSubprocVecEnv`` 简易冒烟测试脚本。

依次实例化两种向量化环境包装器，打印基本信息，并按顺序调用类的所有
公开方法，在终端输出每一步的执行结果，便于人工检查是否有错误。

运行方式::

    python -m tests.test_wrappers
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from src.envs.mamujoco.multi_task import MultiTaskMaMuJoCo
from src.wrappers import ShareDummyVecEnv, ShareSubprocVecEnv

# 测试配置文件路径与并行环境数量
_CONFIG_PATH = Path("configs") / "mamujoco" / "test.yaml"
_NUM_ENVS = 2


def _run_step(
    title: str,
    func: Callable[[], Any],
) -> None:
    """
    执行单个测试步骤并打印结果。

    参数:
        title: 步骤标题，将在终端中显示。
        func: 无参可调用对象，执行实际操作并返回任意结果。
    """
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
    if isinstance(value, tuple):
        return tuple(_summarize(item) for item in value)
    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
        return [_summarize(item) for item in value]
    return value


def _load_config(
    path: Path,
) -> dict[str, Any]:
    """从指定 YAML 文件加载总线环境配置。"""
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _make_env_factory(
    domains_config: dict[str, Any],
) -> Callable[[], MultiTaskMaMuJoCo]:
    """
    构造一个无参环境工厂，便于多次创建独立的总线环境实例。

    参数:
        domains_config: ``MultiTaskMaMuJoCo`` 所需的 ``domains`` 字典。

    返回:
        无参可调用对象，每次调用返回一个新的 ``MultiTaskMaMuJoCo`` 实例。
    """

    def factory() -> MultiTaskMaMuJoCo:
        return MultiTaskMaMuJoCo(domains=domains_config)

    return factory


def _sample_actions(
    vec_env: ShareDummyVecEnv | ShareSubprocVecEnv,
) -> np.ndarray:
    """
    采样符合统一动作空间的随机动作 batch。

    返回:
        形状 ``(num_envs, n_agents, max_action_dim)`` 的随机动作数组。
    """
    actions_per_env = []
    for _ in range(vec_env.num_envs):
        actions_per_env.append(
            np.stack([space.sample() for space in vec_env.action_space]),
        )
    return np.stack(actions_per_env)


def _exercise_vec_env(
    vec_env: ShareDummyVecEnv | ShareSubprocVecEnv,
    label: str,
) -> None:
    """
    依次调用一个向量化环境的全部公开方法。

    参数:
        vec_env: 已实例化的向量化环境对象。
        label: 用于在终端区分日志的标签（"Dummy" / "Subproc"）。
    """
    print("\n" + "-" * 60)
    print(f"[{label}] 基本信息")
    print("-" * 60)
    print(f"  类名                     : {type(vec_env).__name__}")
    print(f"  num_envs                 : {vec_env.num_envs}")
    print(f"  n_agents                 : {vec_env.n_agents}")
    print(f"  observation_space[0]     : {vec_env.observation_space[0]}")
    print(f"  shared_observation_space : {vec_env.shared_observation_space[0]}")
    print(f"  action_space[0]          : {vec_env.action_space[0]}")

    _run_step(
        f"[{label}] reset()",
        lambda: vec_env.reset(),
    )

    _run_step(
        f"[{label}] step(随机动作)",
        lambda: vec_env.step(_sample_actions(vec_env)),
    )

    _run_step(
        f"[{label}] reset_task('run')",
        lambda: vec_env.reset_task("run"),
    )
    _run_step(
        f"[{label}] reset() after switch",
        lambda: vec_env.reset(),
    )
    _run_step(
        f"[{label}] step() after switch",
        lambda: vec_env.step(_sample_actions(vec_env)),
    )

    _run_step(
        f"[{label}] reset_task(0)",
        lambda: vec_env.reset_task(0),
    )
    _run_step(
        f"[{label}] reset_task(None) (随机)",
        lambda: vec_env.reset_task(None),
    )

    _run_step(
        f"[{label}] get_action_mask().shape",
        lambda: vec_env.get_action_mask().shape,
    )
    _run_step(
        f"[{label}] get_task_names()",
        lambda: vec_env.get_task_names(),
    )

    if isinstance(vec_env, ShareSubprocVecEnv):
        _run_step(
            f"[{label}] reset_task_per_env([0, 1])",
            lambda: vec_env.reset_task_per_env(
                list(range(vec_env.num_envs)),
            ),
        )

    _run_step(
        f"[{label}] close()",
        lambda: vec_env.close(),
    )


def main() -> None:
    """脚本入口：依次测试两种向量化环境包装器。"""
    print("=" * 60)
    print("ShareDummyVecEnv / ShareSubprocVecEnv 冒烟测试")
    print("=" * 60)

    print(f"\n[0] 加载配置文件: {_CONFIG_PATH}")
    config = _load_config(_CONFIG_PATH)
    print(f"  配置内容: {config}")

    env_factory = _make_env_factory(config["domains"])
    env_fns = [env_factory for _ in range(_NUM_ENVS)]

    # ------------------------------------------------------------------
    # 1. 单进程版
    # ------------------------------------------------------------------
    print("\n[1] 创建 ShareDummyVecEnv ...")
    dummy_env = ShareDummyVecEnv(env_fns)
    _exercise_vec_env(dummy_env, label="Dummy")

    # ------------------------------------------------------------------
    # 2. 多进程版
    # ------------------------------------------------------------------
    print("\n[2] 创建 ShareSubprocVecEnv ...")
    subproc_env = ShareSubprocVecEnv(env_fns)
    _exercise_vec_env(subproc_env, label="Subproc")

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
