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
class CommonConfig:
    """各任务共用参数。"""

    # 站立时躯干最低高度（米），qpos[1] 应高于此值
    stand_height: float = 1.2
    # 站立/行走时直立容许角度偏差（弧度），约 15°
    upright_angle_bound: float = float(np.deg2rad(15))
    # 动作惩罚 margin
    control_margin: float = 1.0


@dataclass(frozen=True)
class StandConfig:
    """站立任务参数。"""

    # 水平速度惩罚 margin（m/s），鼓励静止
    velocity_margin: float = 0.5


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 目标行走速度（m/s）
    speed: float = 1.5
    # 行走时躯干最低高度（米），略低于站立
    min_height: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""

    # 目标奔跑速度（m/s）
    speed: float = 6.0
    # 奔跑时躯干最低高度（米），允许飞行相高度波动
    min_height: float = 0.9
    # 奔跑时直立容许角度（弧度），约 20°，
    # 比行走略宽松，允许自然前倾
    upright_angle_bound: float = float(np.deg2rad(20))


# 全局默认配置实例
_COMMON = CommonConfig()
_STAND = StandConfig()
_WALK = WalkConfig()
_RUN = RunConfig()


class Walker2dMultiTask(MultiAgentMujocoEnv):
    """
    Walker2d 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Walker2d，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    奖励设计均以躯干直立和高度为基础约束，确保
    智能体在行走和奔跑时保持类人姿态，避免
    以爬行、滑行或翻转等非自然方式完成运动任务。

    支持的任务集:
        - stand: 站立保持平衡
        - walk_fwd: 正向行走
        - walk_bwd: 反向行走
        - run_fwd: 正向奔跑
        - run_bwd: 反向奔跑
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
                2 个智能体各控制 3 个关节。
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
            pitch_deg = float(
                np.rad2deg(self._get_torso_pitch())
            )
            ctrl = self._get_control_magnitude()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"height={height:.2f}  "
                f"pitch={pitch_deg:+6.1f}°  "
                f"ctrl={ctrl:.2f}  "
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

    def _get_torso_height(self) -> float:
        """
        获取躯干高度（rootz 关节位置）。

        Walker2d 的 qpos[1] 即 rootz 滑动关节，
        表示躯干在世界坐标系中的 z 位置。

        返回:
            躯干 z 坐标（米）。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[1]
        )

    def _get_torso_pitch(self) -> float:
        """
        获取躯干俯仰角（rooty 关节位置）。

        返回:
            躯干俯仰角（弧度），0 表示竖直，
            正值为前倾，负值为后仰。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[2]
        )

    def _get_control_magnitude(self) -> float:
        """
        获取当前动作信号的平均绝对值。

        返回:
            控制信号的均值幅度。
        """
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        return float(np.mean(np.abs(ctrl)))

    # ------------------------------------------------------------------
    # 共用子奖励
    # ------------------------------------------------------------------

    def _upright_reward(
        self,
        angle_bound: float = _COMMON.upright_angle_bound,
    ) -> float:
        """
        躯干直立子奖励。

        俯仰角在 [-angle_bound, +angle_bound] 内时
        返回 1，超出后线性衰减至 0；确保智能体在
        运动时保持类人直立姿态，而非以爬行、滑行
        或翻转等非自然方式运动。

        参数:
            angle_bound: 直立容许角度偏差上界（弧度）。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return tolerance(
            self._get_torso_pitch(),
            bounds=(-angle_bound, angle_bound),
            margin=angle_bound,
            value_at_margin=0,
            sigmoid="linear",
        )

    def _height_reward(
        self,
        min_height: float = _COMMON.stand_height,
    ) -> float:
        """
        躯干高度子奖励。

        躯干高度达到 min_height 时返回 1，低于时
        gaussian 衰减，防止爬行或蹲伏运动。

        参数:
            min_height: 最低目标高度（米）。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return tolerance(
            self._get_torso_height(),
            bounds=(min_height, float("inf")),
            margin=min_height / 4,
        )

    def _small_control_reward(self) -> float:
        """
        动作平滑子奖励。

        对每个执行器的控制信号用 quadratic sigmoid
        衰减，取均值后压缩到 [0.8, 1.0] 区间，
        避免主导奖励但能有效抑制高频震颤，
        鼓励自然协调的运动模式。

        返回:
            [0.8, 1.0] 区间内的奖励值。
        """
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        raw = float(
            tolerance(
                ctrl,
                margin=_COMMON.control_margin,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
        )
        # 压缩到 [0.8, 1.0]，避免惩罚过重
        return (4 + raw) / 5

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
        站立奖励。

        综合四个子奖励:
            - upright: 躯干保持竖直（±15° 内满分）
            - height: 躯干高度 >= 1.2m
            - small_control: 动作平滑，抑制震颤
            - dont_move: 水平速度接近零，鼓励静止

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        dont_move = tolerance(
            vx,
            bounds=(
                -_STAND.velocity_margin,
                _STAND.velocity_margin,
            ),
            margin=_STAND.velocity_margin,
            value_at_margin=0,
            sigmoid="linear",
        )

        return (
            self._upright_reward()
            * self._height_reward()
            * self._small_control_reward()
            * dont_move
        )

    def _walk_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向行走奖励。

        综合四个子奖励:
            - upright: 躯干保持竖直（±15° 内满分），
                确保以类人直立姿态行走
            - height: 躯干高度 >= 1.0m，防止蹲伏
            - small_control: 动作平滑，鼓励自然步态
            - move: 沿 x 正方向达到 1.5m/s 目标速度，
                压缩到 [1/6, 1] 保证速度不够时仍有
                站立激励

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(_WALK.speed, float("inf")),
            margin=_WALK.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        # 压缩到 [1/6, 1]，速度为零时仍有站立激励
        move = (5 * move + 1) / 6

        return (
            self._upright_reward()
            * self._height_reward(_WALK.min_height)
            * self._small_control_reward()
            * move
        )

    def _walk_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向行走奖励。

        综合四个子奖励:
            - upright: 躯干保持竖直（±15° 内满分）
            - height: 躯干高度 >= 1.0m
            - small_control: 动作平滑
            - move: 沿 x 负方向达到 1.5m/s 目标速度

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(-float("inf"), -_WALK.speed),
            margin=_WALK.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        return (
            self._upright_reward()
            * self._height_reward(_WALK.min_height)
            * self._small_control_reward()
            * move
        )

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向奔跑奖励。

        综合四个子奖励:
            - upright: 躯干保持竖直（±20° 内满分），
                比行走略宽松，允许自然前倾
            - height: 躯干高度 >= 0.9m，允许飞行相
                时的高度波动
            - small_control: 动作平滑
            - move: 沿 x 正方向达到 6.0m/s 目标速度

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        return (
            self._upright_reward(
                _RUN.upright_angle_bound,
            )
            * self._height_reward(_RUN.min_height)
            * self._small_control_reward()
            * move
        )

    def _run_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向奔跑奖励。

        综合四个子奖励:
            - upright: 躯干保持竖直（±20° 内满分）
            - height: 躯干高度 >= 0.9m
            - small_control: 动作平滑
            - move: 沿 x 负方向达到 6.0m/s 目标速度

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(-float("inf"), -_RUN.speed),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6

        return (
            self._upright_reward(
                _RUN.upright_angle_bound,
            )
            * self._height_reward(_RUN.min_height)
            * self._small_control_reward()
            * move
        )
