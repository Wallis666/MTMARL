"""
HalfCheetah 多任务多智能体环境。

基于 Gymnasium-Robotics 提供的 MaMuJoCo（``mamujoco_v1``）封装单智能体
HalfCheetah，并在其之上叠加 5 个任务的奖励整形：

* ``run``           : 向前快跑。
* ``run_backwards`` : 向后快跑。
* ``jump``          : 双脚同时抬高跳跃。
* ``run_front``     : 后脚抬起，仅靠前脚支撑前进。
* ``run_back``      : 前脚抬起，仅靠后脚支撑前进。

所有奖励均通过 ``src.utils.reward.tolerance`` 软指示函数构造，可直接用于
多任务 / Meta-RL / 多智能体强化学习训练。

参考文档:
    https://robotics.farama.org/envs/MaMuJoCo/
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium_robotics import mamujoco_v1
from pettingzoo.utils.env import ParallelEnv

from src.utils.reward import tolerance

# 各任务的默认目标参数
_RUN_SPEED: float = 10.0
_RUN_BACKWARDS_SPEED: float = 8.0
_RUN_ONE_FOOT_SPEED: float = 6.0
_JUMP_SPEED: float = 0.5
_JUMP_HEIGHT: float = 1.2

# 支持的任务名称列表
TASKS: tuple[str, ...] = (
    "run",
    "run_backwards",
    "jump",
    "run_front",
    "run_back",
)


class HalfCheetahMultiTask(ParallelEnv):
    """
    多任务版 HalfCheetah 多智能体环境。

    本类是对 ``mamujoco_v1.parallel_env`` 的薄包装，遵循 PettingZoo 的
    ``ParallelEnv`` 接口（``reset`` / ``step`` / ``close``），并在 ``step``
    返回的奖励字典上覆盖为当前任务对应的多任务奖励。
    """

    metadata = {"name": "half_cheetah_multi_task_v0"}

    def __init__(
        self,
        agent_conf: str = "2x3",
        run_speed: float = _RUN_SPEED,
        run_backwards_speed: float = _RUN_BACKWARDS_SPEED,
        run_one_foot_speed: float = _RUN_ONE_FOOT_SPEED,
        jump_speed: float = _JUMP_SPEED,
        jump_height: float = _JUMP_HEIGHT,
        default_task: str = "run",
        **env_kwargs: Any,
    ) -> None:
        """
        初始化多任务 HalfCheetah 环境。

        参数:
            agent_conf: MaMuJoCo 智能体划分方式，例如 ``"2x3"`` 表示
                将 6 个关节均分给 2 个智能体，每个控制 3 个。
            run_speed: ``run`` 任务期望达到的最低前向速度。
            run_backwards_speed: ``run_backwards`` 任务期望达到的最低反向速度。
            run_one_foot_speed: ``run_front`` / ``run_back`` 任务期望达到
                的最低前向速度。
            jump_speed: ``jump`` 任务允许的最大水平速度（要求接近原地）。
            jump_height: ``jump`` 与单脚跑任务期望达到的最低身体高度。
            default_task: 环境创建后默认激活的任务名。
            **env_kwargs: 透传给 ``mamujoco_v1.parallel_env`` 的额外参数。

        异常:
            ValueError: 当 ``default_task`` 不在 ``TASKS`` 中时抛出。
        """
        if default_task not in TASKS:
            raise ValueError(
                f"未知的任务名 {default_task!r}，可选: {TASKS}。"
            )

        # 创建底层 MaMuJoCo 并行环境
        self._env = mamujoco_v1.parallel_env(
            scenario="HalfCheetah",
            agent_conf=agent_conf,
            **env_kwargs,
        )

        # PettingZoo 标准属性透传
        self.possible_agents = list(self._env.possible_agents)
        self.agents = list(self._env.agents)
        self.observation_spaces = self._env.observation_spaces
        self.action_spaces = self._env.action_spaces

        # 任务相关参数
        self.run_speed = run_speed
        self.run_backwards_speed = run_backwards_speed
        self.run_one_foot_speed = run_one_foot_speed
        self.jump_speed = jump_speed
        self.jump_height = jump_height

        self._task_idx: int = TASKS.index(default_task)

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv 接口
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """
        重置环境并返回初始观测。

        参数:
            seed: 随机种子。
            options: 透传给底层环境的额外选项。

        返回:
            ``(observations, infos)`` 二元组，键均为智能体名称。
        """
        observations, infos = self._env.reset(seed=seed, options=options)
        self.agents = list(self._env.agents)
        return observations, infos

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """
        执行一步动作并按当前任务计算多任务奖励。

        参数:
            actions: 以智能体名为键的动作字典。

        返回:
            标准的 PettingZoo 并行返回元组
            ``(observations, rewards, terminations, truncations, infos)``，
            其中 ``rewards`` 已被替换为当前任务的整形奖励，所有智能体共享
            同一奖励标量。
        """
        observations, _, terminations, truncations, infos = self._env.step(
            actions,
        )
        self.agents = list(self._env.agents)

        task_reward = self._compute_task_reward(infos)
        rewards = {agent: task_reward for agent in self.possible_agents}

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        """关闭底层环境，释放资源。"""
        self._env.close()

    def observation_space(self, agent: str):
        """返回指定智能体的观测空间。"""
        return self._env.observation_space(agent)

    def action_space(self, agent: str):
        """返回指定智能体的动作空间。"""
        return self._env.action_space(agent)

    # ------------------------------------------------------------------
    # 多任务接口
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> tuple[str, ...]:
        """返回所有支持的任务名称。"""
        return TASKS

    @property
    def n_tasks(self) -> int:
        """返回支持的任务数量。"""
        return len(TASKS)

    @property
    def task(self) -> str:
        """返回当前激活的任务名。"""
        return TASKS[self._task_idx]

    @property
    def task_idx(self) -> int:
        """返回当前激活任务的索引。"""
        return self._task_idx

    def reset_task(
        self,
        task: str,
    ) -> int:
        """
        切换当前激活任务。

        参数:
            task: 目标任务名，必须属于 ``TASKS``。

        返回:
            切换后的任务索引。

        异常:
            ValueError: 当任务名不在 ``TASKS`` 中时抛出。
        """
        if task not in TASKS:
            raise ValueError(f"未知的任务名 {task!r}，可选: {TASKS}。")
        self._task_idx = TASKS.index(task)
        return self._task_idx

    # ------------------------------------------------------------------
    # 奖励函数实现
    # ------------------------------------------------------------------

    def _compute_task_reward(
        self,
        infos: dict[str, dict[str, Any]],
    ) -> float:
        """
        根据当前任务派发到对应的奖励函数。

        参数:
            infos: ``step`` 返回的 info 字典，从中读取速度信息。

        返回:
            当前任务下的标量奖励值。

        异常:
            NotImplementedError: 当任务名未实现时抛出。
        """
        x_velocity = self._extract_x_velocity(infos)

        match self.task:
            case "run":
                return self._run_reward(x_velocity)
            case "run_backwards":
                return self._run_backwards_reward(x_velocity)
            case "jump":
                return self._jump_reward(x_velocity)
            case "run_front":
                return self._run_one_foot_reward(x_velocity, "bfoot")
            case "run_back":
                return self._run_one_foot_reward(x_velocity, "ffoot")
            case _:
                raise NotImplementedError(
                    f"任务 {self.task!r} 尚未实现。"
                )

    def _run_reward(
        self,
        x_velocity: float,
    ) -> float:
        """计算 ``run`` 任务奖励：前向速度越接近 ``run_speed`` 越高。"""
        return float(
            tolerance(
                x_velocity,
                bounds=(self.run_speed, float("inf")),
                margin=self.run_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        )

    def _run_backwards_reward(
        self,
        x_velocity: float,
    ) -> float:
        """计算 ``run_backwards`` 任务奖励：反向速度越大越高。"""
        return float(
            tolerance(
                -x_velocity,
                bounds=(self.run_backwards_speed, float("inf")),
                margin=self.run_backwards_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        )

    def _jump_reward(
        self,
        x_velocity: float,
    ) -> float:
        """计算 ``jump`` 任务奖励：双脚同时抬高且原地不动。"""
        front_reward = self._stand_one_foot_reward(x_velocity, "ffoot")
        back_reward = self._stand_one_foot_reward(x_velocity, "bfoot")
        return 0.5 * (front_reward + back_reward)

    def _stand_one_foot_reward(
        self,
        x_velocity: float,
        which_foot: str,
    ) -> float:
        """
        单脚原地抬高奖励。

        参数:
            x_velocity: 当前前向速度。
            which_foot: 计算高度时考虑的脚，``"ffoot"`` 或 ``"bfoot"``。

        返回:
            高度奖励占主导（权重 5）、速度奖励占次（权重 1）的加权值。
        """
        speed_reward = tolerance(
            x_velocity,
            bounds=(-self.jump_speed, self.jump_speed),
            margin=self.jump_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        torso_height = self._get_body_z("torso")
        foot_height = self._get_body_z(which_foot)
        height = 0.5 * (torso_height + foot_height)
        height_reward = tolerance(
            height,
            bounds=(self.jump_height, float("inf")),
            margin=0.5 * self.jump_height,
        )

        return float((5 * height_reward + speed_reward) / 6)

    def _run_one_foot_reward(
        self,
        x_velocity: float,
        which_foot: str,
    ) -> float:
        """
        单脚抬起前进奖励。

        参数:
            x_velocity: 当前前向速度。
            which_foot: 需要抬离地面的那只脚。

        返回:
            ``up_reward`` 与速度奖励的乘积，鼓励"先抬起脚再跑"的行为。
        """
        torso_height = self._get_body_z("torso")
        torso_up = tolerance(
            torso_height,
            bounds=(self.jump_height, float("inf")),
            margin=0.5 * self.jump_height,
        )

        foot_height = self._get_body_z(which_foot)
        foot_up = tolerance(
            foot_height,
            bounds=(self.jump_height, float("inf")),
            margin=0.5 * self.jump_height,
        )
        up_reward = (3 * foot_up + 2 * torso_up) / 5

        speed_reward = tolerance(
            x_velocity,
            bounds=(self.run_one_foot_speed, float("inf")),
            margin=self.run_one_foot_speed,
        )

        return float(up_reward * (5 * speed_reward + 1) / 6)

    # ------------------------------------------------------------------
    # 底层 MuJoCo 状态访问
    # ------------------------------------------------------------------

    def _extract_x_velocity(
        self,
        infos: dict[str, dict[str, Any]],
    ) -> float:
        """
        从 ``info`` 字典中提取躯干前向速度。

        参数:
            infos: ``step`` 返回的 info 字典。

        返回:
            前向（x 方向）速度，单位 m/s；若 info 中无该字段则
            从底层 MuJoCo data 中读取。
        """
        for agent_info in infos.values():
            if "x_velocity" in agent_info:
                return float(agent_info["x_velocity"])

        single_env = self._env.single_agent_env.unwrapped
        return float(single_env.data.qvel[0])

    def _get_body_z(
        self,
        body_name: str,
    ) -> float:
        """
        读取指定 body 的 z 轴坐标。

        参数:
            body_name: MuJoCo 模型中 body 的名称，例如 ``"torso"``、
                ``"ffoot"``、``"bfoot"``。

        返回:
            该 body 当前的 z 坐标（高度），单位 m。
        """
        single_env = self._env.single_agent_env.unwrapped
        return float(single_env.data.body(body_name).xpos[2])
