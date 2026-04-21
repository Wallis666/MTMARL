"""
Swimmer 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供多种游泳任务的自定义奖励函数，支持在任务间动态切换。
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
class SwimFwdConfig:
    """正向游泳任务参数。"""

    # 目标速度（m/s）
    speed: float = 2.0


@dataclass(frozen=True)
class SwimBwdConfig:
    """反向游泳任务参数。"""

    # 目标速度（m/s）
    speed: float = 1.5


# 全局默认配置实例
_SWIM_FWD = SwimFwdConfig()
_SWIM_BWD = SwimBwdConfig()


# ------------------------------------------------------------------
# 环境类
# ------------------------------------------------------------------


class SwimmerMultiTask(MultiAgentMujocoEnv):
    """
    Swimmer 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Swimmer，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - swim_fwd: 正向游泳
        - swim_bwd: 反向游泳
    """

    TASKS: list[str] = [
        "swim_fwd",
        "swim_bwd",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Swimmer 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "2x1" 表示
                2 个智能体各控制 1 个关节。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        super().__init__(
            scenario="Swimmer",
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
        task_reward = self._compute_reward(infos)
        rewards = {agent: task_reward for agent in obs}
        # 仅在 human 渲染模式下打印，不影响训练
        if self._render_mode == "human":
            vx = self._get_x_velocity(infos)
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        return obs, rewards, terms, truncs, infos

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _get_x_velocity(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        从信息字典中提取 x 方向速度。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            x 方向线速度。
        """
        # 所有智能体共享同一底层环境，取任意一个即可
        info = next(iter(infos.values()))
        return float(info.get("x_velocity", 0.0))

    # ------------------------------------------------------------------
    # 奖励分发
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        根据当前任务计算奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            当前任务对应的标量奖励值。

        异常:
            NotImplementedError: 当前任务未实现时抛出。
        """
        task = self.task
        if task == "swim_fwd":
            return self._swim_fwd_reward(infos)
        elif task == "swim_bwd":
            return self._swim_bwd_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _swim_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向游泳奖励。

        鼓励智能体沿 x 正方向达到目标速度。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        speed = self._get_x_velocity(infos)
        return tolerance(
            speed,
            bounds=(_SWIM_FWD.speed, float("inf")),
            margin=_SWIM_FWD.speed,
            value_at_margin=0,
            sigmoid="linear",
        )

    def _swim_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向游泳奖励。

        鼓励智能体沿 x 负方向达到目标速度。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        speed = self._get_x_velocity(infos)
        return tolerance(
            speed,
            bounds=(-float("inf"), -_SWIM_BWD.speed),
            margin=_SWIM_BWD.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
