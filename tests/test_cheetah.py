"""
HalfCheetahMultiTask 冒烟测试脚本。

直接 ``python tests/test_cheetah.py`` 运行，用于验证：
  * 单智能体 (1x6) 与多智能体 (2x3 / 6x1) 配置下能否正常创建；
  * PettingZoo ParallelEnv 接口 (reset/step/close/spaces) 是否可用；
  * 多任务接口 (tasks/n_tasks/task/task_idx/reset_task) 是否正确；
  * 5 个任务在切换后均能产出有限标量奖励，且所有智能体共享同一标量；
  * 底层 MuJoCo 状态访问 (_get_body_z / _extract_x_velocity / _upright_reward)。
"""

from __future__ import annotations

import math
import traceback

import numpy as np

from src.envs.mamujoco.tasks.cheetah import TASKS, HalfCheetahMultiTask


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  - {msg}")


def _sample_actions(env: HalfCheetahMultiTask) -> dict[str, np.ndarray]:
    return {
        agent: env.action_space(agent).sample() for agent in env.agents
    }


def run_config(agent_conf: str | None) -> None:
    label = agent_conf if agent_conf is not None else "single(None)"
    print(f"\n=== 配置 agent_conf={label} ===")

    env = HalfCheetahMultiTask(agent_conf=agent_conf)
    try:
        # --- 基本属性 ---
        _check(len(env.possible_agents) >= 1, "possible_agents 非空")
        _check(env.n_tasks == len(TASKS) == 5, "n_tasks == 5")
        _check(env.tasks == TASKS, "tasks 元组一致")
        _check(env.task == "run" and env.task_idx == 0, "默认任务为 run")

        # --- spaces ---
        for ag in env.possible_agents:
            obs_sp = env.observation_space(ag)
            act_sp = env.action_space(ag)
            _check(obs_sp.shape is not None, f"{ag} 观测空间形状存在")
            _check(act_sp.shape is not None, f"{ag} 动作空间形状存在")

        # --- reset ---
        obs, infos = env.reset(seed=0)
        _check(set(obs.keys()) == set(env.agents), "reset 观测键覆盖所有 agents")
        for ag, o in obs.items():
            _check(
                isinstance(o, np.ndarray)
                and o.shape == env.observation_space(ag).shape,
                f"{ag} 观测形状匹配",
            )

        # --- step + 五个任务奖励 ---
        for task in TASKS:
            idx = env.reset_task(task)
            _check(env.task == task and env.task_idx == idx,
                   f"reset_task({task}) 切换成功")
            env.reset(seed=1)
            obs, rewards, terms, truncs, infos = env.step(_sample_actions(env))

            _check(set(rewards.keys()) == set(env.possible_agents),
                   f"[{task}] rewards 键覆盖 possible_agents")
            vals = list(rewards.values())
            _check(all(math.isfinite(v) for v in vals),
                   f"[{task}] rewards 全部有限")
            _check(len(set(vals)) == 1,
                   f"[{task}] 所有 agent 共享同一标量奖励")
            _check(set(terms.keys()) == set(env.agents)
                   and set(truncs.keys()) == set(env.agents),
                   f"[{task}] terminations/truncations 键正确")

        # --- 内部 helper ---
        x_vel = env._extract_x_velocity(infos)
        _check(math.isfinite(x_vel), "_extract_x_velocity 返回有限值")
        for body in ("torso", "ffoot", "bfoot"):
            z = env._get_body_z(body)
            _check(math.isfinite(z), f"_get_body_z({body}) 有限")
        up = env._upright_reward()
        _check(0.0 <= up <= 1.0, "_upright_reward 在 [0,1]")

        # --- 错误任务名 ---
        try:
            env.reset_task("nope")
        except ValueError:
            print("  ok  - reset_task 非法名抛 ValueError")
        else:
            raise AssertionError("reset_task 未对非法名抛错")
    finally:
        env.close()


def main() -> None:
    failures = 0
    # 单智能体（整只 cheetah 一个 agent）
    for conf in (None, "2x3", "6x1"):
        try:
            run_config(conf)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  FAIL [{conf}]: {e}")
            traceback.print_exc()

    # 非法默认任务
    try:
        HalfCheetahMultiTask(default_task="bogus")
    except ValueError:
        print("\nok  - 非法 default_task 抛 ValueError")
    else:
        failures += 1
        print("\nFAIL - 非法 default_task 未抛错")

    print(f"\n========== 失败数: {failures} ==========")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
