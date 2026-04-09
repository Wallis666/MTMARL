"""
Humanoid 多任务多智能体环境。

基于 Gymnasium-Robotics 提供的 MaMuJoCo（``mamujoco_v1``）封装单智能体
Humanoid，并在其之上叠加 3 个任务的奖励整形：

* ``stand`` : 站立——躯干高度足够且朝向竖直向上。
* ``walk``  : 行走——质心线速度达到行走目标。
* ``run``   : 奔跑——质心线速度达到更高的奔跑目标。

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
_STAND_HEIGHT: float = 1.4
_WALK_SPEED: float = 3.0
_RUN_SPEED: float = 5.0

# 支持的任务名称列表
TASKS: tuple[str, ...] = ("stand", "walk", "run")


class HumanoidMultiTask(ParallelEnv):
    """
    多任务版 Humanoid 多智能体环境。

    本类是对 ``mamujoco_v1.parallel_env`` 的薄包装，遵循 PettingZoo 的
    ``ParallelEnv`` 接口（``reset`` / ``step`` / ``close``），并在 ``step``
    返回的奖励字典上覆盖为当前任务对应的多任务奖励。
    """

    metadata = {"name": "humanoid_multi_task_v0"}

    def __init__(
        self,
        agent_conf: str = "9|8",
        stand_height: float = _STAND_HEIGHT,
        walk_speed: float = _WALK_SPEED,
        run_speed: float = _RUN_SPEED,
        default_task: str = "stand",
        **env_kwargs: Any,
    ) -> None:
        """
        初始化多任务 Humanoid 环境。

        参数:
            agent_conf: MaMuJoCo 智能体划分方式，例如 ``"9|8"`` 表示
                把 17 个关节按 9/8 拆给两个智能体。
            stand_height: ``stand`` 任务期望达到的最低躯干高度，单位 m。
            walk_speed: ``walk`` 任务期望达到的最低质心线速度。
            run_speed: ``run`` 任务期望达到的最低质心线速度。
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
            scenario="Humanoid",
            agent_conf=agent_conf,
            **env_kwargs,
        )

        # PettingZoo 标准属性透传
        self.possible_agents = list(self._env.possible_agents)
        self.agents = list(self._env.agents)
        self.observation_spaces = self._env.observation_spaces
        self.action_spaces = self._env.action_spaces

        # 任务相关参数
        self.stand_height = stand_height
        self.walk_speed = walk_speed
        self.run_speed = run_speed

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
        forward_velocity = self._extract_forward_velocity(infos)

        match self.task:
            case "stand":
                return self._stand_reward()
            case "walk":
                return self._walk_reward(forward_velocity)
            case "run":
                return self._run_reward(forward_velocity)
            case _:
                raise NotImplementedError(
                    f"任务 {self.task!r} 尚未实现。"
                )

    def _stand_reward(self) -> float:
        """
        计算 ``stand`` 任务奖励：躯干够高且竖直向上。

        返回:
            高度奖励占主导（权重 3）、朝上度占次（权重 1）的加权值。
        """
        standing = tolerance(
            self._get_torso_height(),
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height / 2,
        )
        upright = (1.0 + self._get_torso_upright()) / 2.0
        return float((3 * standing + upright) / 4)

    def _walk_reward(
        self,
        forward_velocity: float,
    ) -> float:
        """
        计算 ``walk`` 任务奖励：质心线速度达到 ``walk_speed``。

        参数:
            forward_velocity: 当前前向线速度。

        返回:
            采用宽松线性 sigmoid（``value_at_margin=0.5``）的速度奖励。
        """
        return float(
            tolerance(
                forward_velocity,
                bounds=(self.walk_speed, float("inf")),
                margin=self.walk_speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
        )

    def _run_reward(
        self,
        forward_velocity: float,
    ) -> float:
        """
        计算 ``run`` 任务奖励：质心线速度达到 ``run_speed``。

        参数:
            forward_velocity: 当前前向线速度。

        返回:
            采用严格线性 sigmoid（``value_at_margin=0``）的速度奖励。
        """
        return float(
            tolerance(
                forward_velocity,
                bounds=(self.run_speed, float("inf")),
                margin=self.run_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        )

    # ------------------------------------------------------------------
    # 底层 MuJoCo 状态访问
    # ------------------------------------------------------------------

    def _extract_forward_velocity(
        self,
        infos: dict[str, dict[str, Any]],
    ) -> float:
        """
        从 ``info`` 字典中提取躯干前向线速度。

        参数:
            infos: ``step`` 返回的 info 字典。

        返回:
            前向（x 方向）质心速度，单位 m/s；若 info 中无对应字段
            则从底层 MuJoCo ``subtree_linvel`` 中读取。
        """
        for agent_info in infos.values():
            if "x_velocity" in agent_info:
                return float(agent_info["x_velocity"])
            if "reward_linvel" in agent_info:
                return float(agent_info["reward_linvel"])

        return float(self._get_center_of_mass_velocity()[0])

    def _get_torso_height(self) -> float:
        """
        读取躯干的 z 轴坐标（高度）。

        返回:
            躯干当前的 z 坐标，单位 m。
        """
        single_env = self._env.single_agent_env.unwrapped
        return float(single_env.data.body("torso").xpos[2])

    def _get_torso_upright(self) -> float:
        """
        读取躯干的"朝上度"。

        返回:
            躯干旋转矩阵中 zz 分量的值，``1`` 表示完全竖直向上，
            ``-1`` 表示完全倒立。
        """
        single_env = self._env.single_agent_env.unwrapped
        return float(single_env.data.body("torso").xmat[8])

    def _get_center_of_mass_velocity(self) -> np.ndarray:
        """
        读取躯干所在子树的线速度向量。

        返回:
            形状为 ``(3,)`` 的线速度向量 ``[vx, vy, vz]``。
        """
        single_env = self._env.single_agent_env.unwrapped
        torso_id = single_env.model.body("torso").id
        return np.asarray(
            single_env.data.subtree_linvel[torso_id],
            dtype=np.float64,
        ).copy()
