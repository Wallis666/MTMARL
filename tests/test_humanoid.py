"""
``HumanoidMultiTask`` 简易冒烟测试脚本。

依次实例化环境、打印基本信息，并按顺序调用类的所有方法，
在终端输出每一步的执行结果，便于人工检查是否有错误。

运行方式::

    python -m tests.test_humanoid
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from src.envs.mamujoco.tasks.humanoid import (
    TASKS,
    HumanoidMultiTask,
)


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


def main() -> None:
    """脚本入口：依次调用 ``HumanoidMultiTask`` 的全部方法。"""
    print("=" * 60)
    print("HumanoidMultiTask 冒烟测试")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 实例化
    # ------------------------------------------------------------------
    print("\n[1] 创建环境实例 ...")
    env = HumanoidMultiTask(agent_conf="9|8")

    # ------------------------------------------------------------------
    # 2. 打印类基本信息
    # ------------------------------------------------------------------
    print("\n[2] 类基本信息：")
    print(f"  类名             : {type(env).__name__}")
    print(f"  metadata         : {env.metadata}")
    print(f"  possible_agents  : {env.possible_agents}")
    print(f"  agents           : {env.agents}")
    print(f"  tasks            : {env.tasks}")
    print(f"  n_tasks          : {env.n_tasks}")
    print(f"  task (默认)      : {env.task}")
    print(f"  task_idx         : {env.task_idx}")
    print(f"  stand_height     : {env.stand_height}")
    print(f"  walk_speed       : {env.walk_speed}")
    print(f"  run_speed        : {env.run_speed}")

    # ------------------------------------------------------------------
    # 3. 依次调用所有公开 / 私有方法
    # ------------------------------------------------------------------

    _run_step(
        "reset(seed=0)",
        lambda: env.reset(seed=0),
    )

    _run_step(
        "observation_space(agent[0])",
        lambda: env.observation_space(env.possible_agents[0]),
    )

    _run_step(
        "action_space(agent[0])",
        lambda: env.action_space(env.possible_agents[0]),
    )

    _run_step(
        "step(随机动作)",
        lambda: env.step(
            {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            },
        ),
    )

    for task in TASKS:
        _run_step(
            f"reset_task({task!r})",
            lambda t=task: env.reset_task(t),
        )

    _run_step(
        "_get_torso_height()",
        lambda: env._get_torso_height(),
    )
    _run_step(
        "_get_torso_upright()",
        lambda: env._get_torso_upright(),
    )
    _run_step(
        "_get_center_of_mass_velocity()",
        lambda: env._get_center_of_mass_velocity(),
    )

    _run_step(
        "_extract_forward_velocity(伪造 x_velocity)",
        lambda: env._extract_forward_velocity(
            {env.possible_agents[0]: {"x_velocity": 1.23}},
        ),
    )
    _run_step(
        "_extract_forward_velocity(伪造 reward_linvel)",
        lambda: env._extract_forward_velocity(
            {env.possible_agents[0]: {"reward_linvel": 2.34}},
        ),
    )
    _run_step(
        "_extract_forward_velocity(空 info, 走兜底)",
        lambda: env._extract_forward_velocity(
            {agent: {} for agent in env.possible_agents},
        ),
    )

    _run_step(
        "_stand_reward()",
        lambda: env._stand_reward(),
    )
    _run_step(
        "_walk_reward(walk_speed)",
        lambda: env._walk_reward(env.walk_speed),
    )
    _run_step(
        "_run_reward(run_speed)",
        lambda: env._run_reward(env.run_speed),
    )

    for task in TASKS:
        env.reset_task(task)
        _run_step(
            f"_compute_task_reward(任务={task})",
            lambda: env._compute_task_reward(
                {env.possible_agents[0]: {"x_velocity": env.walk_speed}},
            ),
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
