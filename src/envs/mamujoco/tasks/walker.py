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
    # 速度容忍范围（m/s），在 margin 内衰减
    speed_margin: float = 1.0
    # 两脚水平间距目标下限（米）
    foot_spread_low: float = 0.25
    # 两脚水平间距目标上限（米）
    foot_spread_high: float = 0.4


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # torso 高度合理区间下限（米）
    height_low: float = 1.1
    # torso 高度合理区间上限（米）
    height_high: float = 1.4
    # 高度 tolerance 的 margin
    height_margin: float = 0.2
    # 早期终止高度阈值（米），低于此值视为倒下
    terminate_height: float = 1.0
    # 目标前进速度（m/s），正值向前，负值向后
    target_speed: float = 1.0
    # 前进速度 tolerance 的 margin
    speed_margin: float = 1.0
    # z 方向速度 tolerance 的 margin（防跳）
    z_velocity_margin: float = 2.0
    # 脚部着地判定高度上限（米）
    foot_ground_height: float = 0.15
    # 脚部着地 tolerance 的 margin
    foot_ground_margin: float = 0.15


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""

    # torso 高度合理区间下限（米）
    height_low: float = 0.9
    # torso 高度合理区间上限（米）
    height_high: float = 1.5
    # 高度 tolerance 的 margin
    height_margin: float = 0.2
    # 早期终止高度阈值（米），低于此值视为倒下
    terminate_height: float = 0.8
    # 目标前进速度（m/s），正值向前，负值向后
    target_speed: float = 5.0
    # 前进速度 tolerance 的 margin
    speed_margin: float = 5.0
    # z 方向速度 tolerance 的 margin（防跳）
    z_velocity_margin: float = 3.0
    # 脚部着地判定高度上限（米）
    foot_ground_height: float = 0.25
    # 脚部着地 tolerance 的 margin
    foot_ground_margin: float = 0.25


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
            vz = self._get_z_velocity()
            right_z, left_z = self._get_foot_heights()
            data = self.single_agent_env.unwrapped.data
            foot_dist = abs(
                data.body("foot").xpos[0]
                - data.body("foot_left").xpos[0]
            )
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"v_z={vz:+5.2f}  "
                f"height={height:.2f}  "
                f"upright={upright:+.2f}  "
                f"foot_dist={foot_dist:.2f}m  "
                f"foot_z=({right_z:.3f}, {left_z:.3f})  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        # 早期终止：walk/run 任务中 torso 高度低于阈值视为倒下
        if self.task in ("walk_fwd", "walk_bwd"):
            if self._get_torso_height() < _WALK.terminate_height:
                terms = {agent: True for agent in terms}
        if self.task in ("run_fwd", "run_bwd"):
            if self._get_torso_height() < _RUN.terminate_height:
                terms = {agent: True for agent in terms}

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

    def _get_z_velocity(self) -> float:
        """
        获取 torso 的 z 方向速度。

        返回:
            rootz 关节的广义速度（m/s）。
        """
        return float(
            self.single_agent_env.unwrapped.data.qvel[1]
        )

    def _get_foot_heights(self) -> tuple[float, float]:
        """
        获取左右脚刚体的 z 坐标。

        返回:
            (右脚 z, 左脚 z) 元组。
        """
        data = self.single_agent_env.unwrapped.data
        return (
            float(data.body("foot").xpos[2]),
            float(data.body("foot_left").xpos[2]),
        )

    def _get_feet_spread(self) -> float:
        """
        评估两脚的水平间距。

        取 foot 和 foot_left 刚体 x 坐标的绝对差值，
        落在 [foot_spread_low, foot_spread_high] 区间内
        时满分。这直接约束了腿要真正张开到脚的位置，
        防止大腿张开但膝盖弯回来的 |> 姿态。

        返回:
            [0, 1] 区间内的评分。
        """
        data = self.single_agent_env.unwrapped.data
        foot_dist = abs(
            data.body("foot").xpos[0]
            - data.body("foot_left").xpos[0]
        )
        return float(tolerance(
            foot_dist,
            bounds=(
                _STAND.foot_spread_low,
                _STAND.foot_spread_high,
            ),
            margin=_STAND.foot_spread_low,
        ))

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
            return self._stand_reward(infos)
        elif task == "walk_fwd":
            return self._walk_fwd_reward(infos)
        elif task == "walk_bwd":
            return self._walk_bwd_reward(infos)
        elif task == "run_fwd":
            return self._run_fwd_reward(infos)
        elif task == "run_bwd":
            return self._run_bwd_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _stand_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        站立任务奖励。

        综合四个因子:
            - standing: torso 高度 ≥ 阈值时满分
            - upright: 躯干直立度映射到 [0, 1]
            - small_velocity: x 速度接近 0 时满分
            - feet_spread: 两脚水平间距在目标区间内时满分
        合并公式: standing × upright × small_velocity × feet_spread

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
        vx = self._get_x_velocity(infos)
        small_velocity = tolerance(
            vx,
            bounds=(0.0, 0.0),
            margin=_STAND.speed_margin,
        )

        # 两脚水平间距
        feet_spread = self._get_feet_spread()

        return (
            standing
            * upright
            * small_velocity
            * feet_spread
        )

    def _walk_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        向前行走任务奖励。

        综合五个因子:
            - standing: torso 高度在合理区间内时满分
            - upright: 躯干直立度映射到 [0, 1]
            - move: 水平速度达到目标速度时满分
            - smooth_z: z 方向速度接近 0 时满分（防跳）
            - grounded: 至少一只脚着地时满分（防飞行）
        合并公式: standing × upright × move × smooth_z × grounded

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 高度在合理区间
        standing = tolerance(
            self._get_torso_height(),
            bounds=(
                _WALK.height_low,
                _WALK.height_high,
            ),
            margin=_WALK.height_margin,
        )

        # 躯干直立
        upright = (1 + 5 * self._get_torso_upright()) / 6

        # 水平速度达到目标（正方向）
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(_WALK.target_speed, float("inf")),
            margin=_WALK.speed_margin,
        )

        # z 方向速度接近 0（防跳）
        vz = self._get_z_velocity()
        smooth_z = tolerance(
            vz,
            bounds=(0.0, 0.0),
            margin=_WALK.z_velocity_margin,
        )

        # 至少一只脚着地（防飞行相）
        right_z, left_z = self._get_foot_heights()
        right_ground = tolerance(
            right_z,
            bounds=(0.0, _WALK.foot_ground_height),
            margin=_WALK.foot_ground_margin,
        )
        left_ground = tolerance(
            left_z,
            bounds=(0.0, _WALK.foot_ground_height),
            margin=_WALK.foot_ground_margin,
        )
        grounded = max(right_ground, left_ground)

        return (
            standing
            * upright
            * move
            * smooth_z
            * grounded
        )

    def _walk_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        向后行走任务奖励。

        综合五个因子:
            - standing: torso 高度在合理区间内时满分
            - upright: 躯干直立度映射到 [0, 1]
            - move: 水平速度达到目标负速度时满分
            - smooth_z: z 方向速度接近 0 时满分（防跳）
            - grounded: 至少一只脚着地时满分（防飞行）
        合并公式: standing × upright × move × smooth_z × grounded

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 高度在合理区间
        standing = tolerance(
            self._get_torso_height(),
            bounds=(
                _WALK.height_low,
                _WALK.height_high,
            ),
            margin=_WALK.height_margin,
        )

        # 躯干直立
        upright = (1 + 5 * self._get_torso_upright()) / 6

        # 水平速度达到目标（负方向）
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(float("-inf"), -_WALK.target_speed),
            margin=_WALK.speed_margin,
        )

        # z 方向速度接近 0（防跳）
        vz = self._get_z_velocity()
        smooth_z = tolerance(
            vz,
            bounds=(0.0, 0.0),
            margin=_WALK.z_velocity_margin,
        )

        # 至少一只脚着地（防飞行相）
        right_z, left_z = self._get_foot_heights()
        right_ground = tolerance(
            right_z,
            bounds=(0.0, _WALK.foot_ground_height),
            margin=_WALK.foot_ground_margin,
        )
        left_ground = tolerance(
            left_z,
            bounds=(0.0, _WALK.foot_ground_height),
            margin=_WALK.foot_ground_margin,
        )
        grounded = max(right_ground, left_ground)

        return (
            standing
            * upright
            * move
            * smooth_z
            * grounded
        )

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        向前奔跑任务奖励。

        综合五个因子:
            - standing: torso 高度在合理区间内时满分
            - upright: 躯干直立度映射到 [0, 1]
            - move: 水平速度达到目标速度时满分
            - smooth_z: z 方向速度接近 0 时满分（防跳）
            - grounded: 至少一只脚着地时满分（防飞行）
        合并公式: standing × upright × move × smooth_z × grounded

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 高度在合理区间
        standing = tolerance(
            self._get_torso_height(),
            bounds=(
                _RUN.height_low,
                _RUN.height_high,
            ),
            margin=_RUN.height_margin,
        )

        # 躯干直立
        upright = (1 + 5 * self._get_torso_upright()) / 6

        # 水平速度达到目标（正方向）
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(_RUN.target_speed, float("inf")),
            margin=_RUN.speed_margin,
        )

        # z 方向速度接近 0（防跳）
        vz = self._get_z_velocity()
        smooth_z = tolerance(
            vz,
            bounds=(0.0, 0.0),
            margin=_RUN.z_velocity_margin,
        )

        # 至少一只脚着地（防飞行相）
        right_z, left_z = self._get_foot_heights()
        right_ground = tolerance(
            right_z,
            bounds=(0.0, _RUN.foot_ground_height),
            margin=_RUN.foot_ground_margin,
        )
        left_ground = tolerance(
            left_z,
            bounds=(0.0, _RUN.foot_ground_height),
            margin=_RUN.foot_ground_margin,
        )
        grounded = max(right_ground, left_ground)

        return (
            standing
            * upright
            * move
            * smooth_z
            * grounded
        )

    def _run_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        向后奔跑任务奖励。

        综合五个因子:
            - standing: torso 高度在合理区间内时满分
            - upright: 躯干直立度映射到 [0, 1]
            - move: 水平速度达到目标负速度时满分
            - smooth_z: z 方向速度接近 0 时满分（防跳）
            - grounded: 至少一只脚着地时满分（防飞行）
        合并公式: standing × upright × move × smooth_z × grounded

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 高度在合理区间
        standing = tolerance(
            self._get_torso_height(),
            bounds=(
                _RUN.height_low,
                _RUN.height_high,
            ),
            margin=_RUN.height_margin,
        )

        # 躯干直立
        upright = (1 + 5 * self._get_torso_upright()) / 6

        # 水平速度达到目标（负方向）
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(float("-inf"), -_RUN.target_speed),
            margin=_RUN.speed_margin,
        )

        # z 方向速度接近 0（防跳）
        vz = self._get_z_velocity()
        smooth_z = tolerance(
            vz,
            bounds=(0.0, 0.0),
            margin=_RUN.z_velocity_margin,
        )

        # 至少一只脚着地（防飞行相）
        right_z, left_z = self._get_foot_heights()
        right_ground = tolerance(
            right_z,
            bounds=(0.0, _RUN.foot_ground_height),
            margin=_RUN.foot_ground_margin,
        )
        left_ground = tolerance(
            left_z,
            bounds=(0.0, _RUN.foot_ground_height),
            margin=_RUN.foot_ground_margin,
        )
        grounded = max(right_ground, left_ground)

        return (
            standing
            * upright
            * move
            * smooth_z
            * grounded
        )
