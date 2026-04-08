"""
``MultiTaskMaMuJoCo`` 简易冒烟测试脚本。

从 ``configs/mamujoco/test.yaml`` 读取域 / 任务配置，依次实例化总线
环境、打印基本信息，并按顺序调用类的所有方法，在终端输出每一步的
执行结果，便于人工检查是否有错误。

运行方式::

    python -m tests.test_multi_task
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from src.envs.mamujoco.multi_task import MultiTaskMaMuJoCo

# 测试配置文件路径
_CONFIG_PATH = Path("configs") / "mamujoco" / "test.yaml"


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
    print(f"  [成功] 返回: {result!r}")


def _load_config(
    path: Path,
) -> dict[str, Any]:
    """
    从指定 YAML 文件加载总线环境配置。

    参数:
        path: YAML 配置文件路径。

    返回:
        解析后的配置字典。
    """
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _sample_actions(
    env: MultiTaskMaMuJoCo,
) -> list[np.ndarray]:
    """根据总线环境的统一动作空间为所有智能体采样动作。"""
    return [space.sample() for space in env.action_space]


def main() -> None:
    """脚本入口：依次调用 ``MultiTaskMaMuJoCo`` 的全部方法。"""
    print("=" * 60)
    print("MultiTaskMaMuJoCo 冒烟测试")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 0. 读取 YAML 配置
    # ------------------------------------------------------------------
    print(f"\n[0] 加载配置文件: {_CONFIG_PATH}")
    config = _load_config(_CONFIG_PATH)
    print(f"  配置内容: {config}")

    # ------------------------------------------------------------------
    # 1. 实例化
    # ------------------------------------------------------------------
    print("\n[1] 创建总线环境实例 ...")
    env = MultiTaskMaMuJoCo(domains=config["domains"])

    # ------------------------------------------------------------------
    # 2. 打印类基本信息
    # ------------------------------------------------------------------
    print("\n[2] 类基本信息：")
    print(f"  类名                 : {type(env).__name__}")
    print(f"  domain_names         : {env.domain_names}")
    print(f"  n_domains            : {env.n_domains}")
    print(f"  task_names           : {env.task_names}")
    print(f"  n_total_tasks        : {env.n_total_tasks}")
    print(f"  task_domain_index    : {env.task_domain_index}")
    print(f"  task_local_index     : {env.task_local_index}")
    print(f"  domain_n_agents      : {env.domain_n_agents}")
    print(f"  max_n_agents         : {env.max_n_agents}")
    print(f"  domain_obs_dims      : {env.domain_obs_dims}")
    print(f"  domain_action_dims   : {env.domain_action_dims}")
    print(f"  observation_dim      : {env.observation_dim}")
    print(f"  max_action_dim       : {env.max_action_dim}")
    print(f"  agent_names          : {env.agent_names}")
    print(f"  current_task         : {env.current_task}")
    print(f"  current_task_index   : {env.current_task_index}")
    print(f"  current_domain_name  : {env.current_domain_name}")

    # ------------------------------------------------------------------
    # 3. 依次调用所有公开 / 私有方法
    # ------------------------------------------------------------------

    _run_step(
        "get_task_names()",
        lambda: env.get_task_names(),
    )

    _run_step(
        "get_action_mask().shape",
        lambda: env.get_action_mask().shape,
    )

    _run_step(
        "reset(seed=0)",
        lambda: [arr.shape for arr in env.reset(seed=0)[0]],
    )

    _run_step(
        "step(随机动作) -> observations 形状",
        lambda: [arr.shape for arr in env.step(_sample_actions(env))[0]],
    )

    _run_step(
        "step(随机动作) -> rewards 数值",
        lambda: [
            float(r[0]) for r in env.step(_sample_actions(env))[2]
        ],
    )

    # 遍历所有任务，依次切换并跑一步
    for task_name in env.task_names:
        _run_step(
            f"reset_task({task_name!r})",
            lambda t=task_name: env.reset_task(t),
        )
        _run_step(
            f"reset() after switch -> obs[0].shape",
            lambda: env.reset(seed=1)[0][0].shape,
        )
        _run_step(
            f"step() in task={task_name}",
            lambda: float(env.step(_sample_actions(env))[2][0][0]),
        )

    # 按下标切换 + 随机切换
    _run_step(
        "reset_task(0) (按下标)",
        lambda: env.reset_task(0),
    )
    _run_step(
        "reset_task(None) (随机)",
        lambda: env.reset_task(None),
    )

    # 当前状态属性
    _run_step(
        "current_domain_index",
        lambda: env.current_domain_index,
    )
    _run_step(
        "current_local_task_index",
        lambda: env.current_local_task_index,
    )
    _run_step(
        "current_env 类型",
        lambda: type(env.current_env).__name__,
    )

    # 内部工具方法
    _run_step(
        "_sync_active_task()",
        lambda: env._sync_active_task(),
    )

    _run_step(
        "_pad_observations(伪造 obs dict)",
        lambda: [
            arr.shape
            for arr in env._pad_observations(
                {
                    agent: np.zeros(
                        env.current_env.observation_space(agent).shape[0],
                        dtype=np.float32,
                    )
                    for agent in env.current_env.possible_agents
                },
            )
        ],
    )

    _run_step(
        "_crop_actions(全零动作)",
        lambda: [
            arr.shape
            for arr in env._crop_actions(
                [
                    np.zeros(env.max_action_dim, dtype=np.float32)
                    for _ in range(env.max_n_agents)
                ],
            )
        ],
    )

    # 异常路径
    _run_step(
        "reset_task('not_exist') 应抛 ValueError",
        lambda: env.reset_task("not_exist"),
    )
    _run_step(
        "reset_task(9999) 应抛 ValueError",
        lambda: env.reset_task(9999),
    )

    _run_step(
        "close()",
        lambda: env.close(),
    )

    print("\n" + "=" * 60)
    print("全部步骤执行完毕")
    print("=" * 60)


if __name__ == "__main__":
    main()
