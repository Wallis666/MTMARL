"""
Walker2d 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供 stand、walk_fwd、walk_bwd、run_fwd、run_bwd 五种任务的
自定义奖励函数，支持在任务间动态切换。
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
class StandConfig:
    """站立任务参数。"""

    # torso 最低站立高度（米）
    height: float = 1.2
    # 目标速度上限（m/s），越接近 0 越好
    max_speed: float = 0.5
    # 两腿夹角下限（度）
    leg_angle_low: float = 30.0
    # 两腿夹角上限（度）
    leg_angle_high: float = 60.0


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""


# 全局默认配置实例
_STAND = StandConfig()
_WALK = WalkConfig()
_RUN = RunConfig()


class Walker2dMultiTask(MultiAgentMujocoEnv):
    """
    Walker2d 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Walker2d，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - stand: 站立保持平衡
        - walk_fwd: 向前行走
        - walk_bwd: 向后行走
        - run_fwd: 向前奔跑
        - run_bwd: 向后奔跑
    """

    TASKS: list[str] = [
        "stand",
        "walk_fwd",
        "walk_bwd",
        "run_fwd",
        "run_bwd",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Walker2d 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "2x3" 表示
                2 个智能体各控制 3 个关节（左右腿）。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        super().__init__(
            scenario="Walker2d",
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
            height = self._get_torso_height()
            upright = self._get_torso_upright()
            leg_ang = self._get_leg_angle()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"height={height:.2f}  "
                f"upright={upright:+.2f}  "
                f"leg_ang={leg_ang:5.1f}°  "
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
        info = next(iter(infos.values()))
        return float(info.get("x_velocity", 0.0))

    def _get_torso_height(self) -> float:
        """
        获取 torso 刚体的绝对 z 轴高度。

        返回:
            torso 的 z 坐标值（米）。
        """
        return float(
            self.single_agent_env.unwrapped.data.body(
                "torso"
            ).xpos[2]
        )

    def _get_torso_upright(self) -> float:
        """
        获取躯干直立度（旋转矩阵 zz 分量）。

        xmat 按行展平为 9 元素数组，xmat[8] 为 z 轴
        在世界 z 方向的投影:
            1.0 表示完全竖直，-1.0 表示完全倒立。

        返回:
            [-1, 1] 区间内的直立度值。
        """
        xmat = (
            self.single_agent_env.unwrapped.data
            .body("torso").xmat
        )
        return float(xmat[8])

    def _get_leg_angle(self) -> float:
        """
        获取左右大腿关节角度之差的绝对值（度）。

        thigh_joint (qpos[3]) 与 thigh_left_joint (qpos[6])
        的差值取绝对值后转为角度。

        返回:
            两腿夹角（度）。
        """
        qpos = self.single_agent_env.unwrapped.data.qpos
        return float(np.degrees(abs(qpos[3] - qpos[6])))

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
        if task == "stand":
            return self._stand_reward()
        elif task == "walk_fwd":
            raise NotImplementedError("walk_fwd 奖励尚未实现")
        elif task == "walk_bwd":
            raise NotImplementedError("walk_bwd 奖励尚未实现")
        elif task == "run_fwd":
            raise NotImplementedError("run_fwd 奖励尚未实现")
        elif task == "run_bwd":
            raise NotImplementedError("run_bwd 奖励尚未实现")
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _stand_reward(self) -> float:
        """
        站立任务奖励。

        综合四个因子:
            - standing: torso 高度 ≥ 阈值时满分
            - upright: 躯干直立度映射到 [0, 1]
            - small_velocity: x 速度接近 0 时满分
            - leg_angle: 两腿夹角在目标区间内时满分
        合并公式: standing × upright × (small_velocity + leg_angle) / 2

        返回:
            [0, 1] 区间内的奖励值。
        """
        standing = tolerance(
            self._get_torso_height(),
            bounds=(_STAND.height, float("inf")),
            margin=_STAND.height / 2,
        )
        upright = (1 + self._get_torso_upright()) / 2

        # 速度越接近 0 越好
        x_vel = float(
            self.single_agent_env.unwrapped.data.qvel[0]
        )
        small_velocity = tolerance(
            x_vel,
            bounds=(-_STAND.max_speed, _STAND.max_speed),
            margin=_STAND.max_speed,
        )

        # 两腿夹角在目标区间内
        leg_angle = self._get_leg_angle()
        leg_angle_reward = tolerance(
            leg_angle,
            bounds=(
                _STAND.leg_angle_low,
                _STAND.leg_angle_high,
            ),
            margin=_STAND.leg_angle_low,
        )

        return (
            standing
            * upright
            * (small_velocity + leg_angle_reward) / 2
        )
