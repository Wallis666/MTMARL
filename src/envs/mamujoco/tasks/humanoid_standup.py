"""
HumanoidStandup 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供 standup 任务的自定义奖励函数，支持在任务间动态切换。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import (
    MultiAgentMujocoEnv,
)
from numpy.typing import NDArray

from src.utils.reward import tolerance


# ------------------------------------------------------------------
# 任务参数配置
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StandupConfig:
    """站起任务参数。"""

    # 目标站立高度（米），即 torso z 坐标（qpos[2]）的期望值
    standup_height: float = 1.5


# 全局默认配置实例
_STANDUP = StandupConfig()


class HumanoidStandupMultiTask(MultiAgentMujocoEnv):
    """
    HumanoidStandup 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为
    HumanoidStandup，并在 step 中用当前任务的自定义奖励
    替换默认奖励。各智能体在同一时间步共享相同的任务奖励
    信号。

    底层环境无终止条件（始终 terminated=False），仅通过
    TimeLimit wrapper 在 1000 步后截断。

    支持的任务集:
        - standup: 从仰卧姿态站起并保持直立
    """

    TASKS: list[str] = [
        "standup",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 HumanoidStandup 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "9|8" 表示
                上半身 9 个关节、下半身 8 个关节。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        super().__init__(
            scenario="HumanoidStandup",
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            render_mode=render_mode,
            **kwargs,
        )

        self._render_mode = render_mode
        self._task_idx: int = 0

    # ------------------------------------------------------------------
    # 任务属性
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """返回当前任务名称。"""
        return self.TASKS[self._task_idx]

    @property
    def task_idx(self) -> int:
        """返回当前任务索引。"""
        return self._task_idx

    @property
    def n_tasks(self) -> int:
        """返回支持的任务总数。"""
        return len(self.TASKS)

    # ------------------------------------------------------------------
    # 任务切换
    # ------------------------------------------------------------------

    def set_task(
        self,
        task: str | int,
    ) -> None:
        """
        切换当前任务。

        参数:
            task: 任务名称（字符串）或任务索引（整数）。

        异常:
            ValueError: 当任务名称不在支持列表中时抛出。
            IndexError: 当任务索引超出范围时抛出。
        """
        if isinstance(task, str):
            if task not in self.TASKS:
                raise ValueError(
                    f"不支持的任务: {task!r}，"
                    f"可选任务: {self.TASKS}"
                )
            self._task_idx = self.TASKS.index(task)
        else:
            if not 0 <= task < len(self.TASKS):
                raise IndexError(
                    f"任务索引 {task} 超出范围"
                    f" [0, {len(self.TASKS)})"
                )
            self._task_idx = int(task)

    # ------------------------------------------------------------------
    # 重写 step：替换奖励
    # ------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, NDArray],
    ) -> tuple[
        dict[str, NDArray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """
        执行一步交互，并用当前任务的奖励替换默认奖励。

        参数:
            actions: 各智能体动作的字典，键为智能体名称。

        返回:
            (观测, 奖励, 终止, 截断, 信息) 五元组。
        """
        obs, _, terms, truncs, infos = super().step(actions)
        task_reward = self._compute_reward()
        rewards = {agent: task_reward for agent in obs}
        # 仅在 human 渲染模式下打印，不影响训练
        if self._render_mode == "human":
            torso_z = self._get_torso_z()
            print(
                f"\rtask={self.task:<8} "
                f"torso={torso_z:.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        return obs, rewards, terms, truncs, infos

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @property
    def _mj_data(self):
        """返回底层 MuJoCo 仿真的 data 对象。"""
        return self.single_agent_env.unwrapped.data

    def _get_torso_z(self) -> float:
        """
        获取 torso 的 z 坐标高度。

        直接读取 qpos[2]（torso 自由关节的世界 z 坐标）。

        返回:
            torso 的 z 坐标值（米）。
        """
        return float(self._mj_data.qpos[2])

    # ------------------------------------------------------------------
    # 奖励分发
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        根据当前任务计算奖励。

        返回:
            当前任务对应的标量奖励值。

        异常:
            NotImplementedError: 当前任务未实现时抛出。
        """
        task = self.task
        if task == "standup":
            return self._standup_reward()
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------
    # TODO
    def _standup_reward(self) -> float:
        """
        站起任务奖励。

        直接读取 qpos[2] 获取 torso z 高度，当高度
        ≥ standup_height 时满分，低于此值按 margin 线性
        衰减至 0。

        返回:
            [0, 1] 区间内的奖励值。
        """
        torso_z = self._get_torso_z()
        return tolerance(
            torso_z,
            bounds=(
                _STANDUP.standup_height, float("inf"),
            ),
            margin=_STANDUP.standup_height,
            value_at_margin=0,
            sigmoid="linear",
        )
