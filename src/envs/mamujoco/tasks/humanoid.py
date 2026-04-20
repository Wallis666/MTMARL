"""
Humanoid 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供多种运动任务的自定义奖励函数，支持在任务间动态切换。
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

    # head 目标高度（米），站立时 head 应达到的高度
    head_target_z: float = 1.4
    # head 高度 margin（米）
    head_z_margin: float = head_target_z / 5
    # 动作惩罚 margin，用于 small_control 子奖励
    control_margin: float = 1.0
    # 躯干最小高度（米），低于此高度判定跌倒并终止 episode
    min_torso_z: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    """正向跑任务参数。"""

    # 目标速度（m/s）
    speed: float = 6.0


@dataclass(frozen=True)
class StandConfig:
    """站立任务参数。"""

    # 水平速度 margin（m/s），
    # gaussian 衰减，同时约束 x 和 y 两轴
    velocity_margin: float = 1.5


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 目标速度（m/s）
    speed: float = 3.0


# 全局默认配置实例
_COMMON = CommonConfig()
_RUN = RunConfig()
_STAND = StandConfig()
_WALK = WalkConfig()


class HumanoidMultiTask(MultiAgentMujocoEnv):
    """
    Humanoid 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Humanoid，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - run_fwd: 正向跑
        - stand: 站立保持平衡
        - walk: 缓慢行走
    """

    TASKS: list[str] = [
        "run_fwd",
        "stand",
        "walk",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Humanoid 多任务环境。

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
            scenario="Humanoid",
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
            torso_z = self._get_body_z("torso")
            head_z = self._get_geom_z("head")
            upright = self._get_torso_upright()
            lfoot_z = self._get_body_z("left_foot")
            rfoot_z = self._get_body_z("right_foot")
            pelvis_z = self._get_body_z("pelvis")
            ctrl = self._get_control_magnitude()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"head={head_z:.2f}  "
                f"torso={torso_z:.2f}  "
                f"pelvis={pelvis_z:.2f}  "
                f"lfoot={lfoot_z:.2f}  "
                f"rfoot={rfoot_z:.2f}  "
                f"upright={upright:+.2f}  "
                f"ctrl={ctrl:.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        # 跌倒早期终止: 躯干高度低于阈值时结束 episode
        if self._has_fallen():
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
        # 所有智能体共享同一底层环境，取任意一个即可
        info = next(iter(infos.values()))
        return float(info.get("x_velocity", 0.0))

    def _get_body_z(
        self,
        body_name: str,
    ) -> float:
        """
        获取指定刚体的 z 轴高度。

        参数:
            body_name: 刚体名称，如 "torso"、"left_foot"。

        返回:
            该刚体当前的 z 坐标值。
        """
        return float(
            self.single_agent_env.unwrapped.data.body(
                body_name
            ).xpos[2]
        )

    def _get_geom_z(
        self,
        geom_name: str,
    ) -> float:
        """
        获取指定几何体的 z 轴高度。

        参数:
            geom_name: 几何体名称，如 "head"。

        返回:
            该几何体当前的 z 坐标值。
        """
        env = self.single_agent_env.unwrapped
        geom_id = env.model.geom(geom_name).id
        return float(env.data.geom_xpos[geom_id][2])

    def _get_control(self) -> NDArray:
        """
        获取当前所有执行器的控制信号。

        返回:
            (n_actuators,) 的控制信号数组。
        """
        return self.single_agent_env.unwrapped.data.ctrl.copy()

    def _get_control_magnitude(self) -> float:
        """
        获取当前动作信号的平均绝对值。

        返回:
            控制信号的均值幅度。
        """
        return float(np.mean(np.abs(self._get_control())))

    def _get_xy_velocity(
        self,
        infos: dict[str, dict],
    ) -> NDArray:
        """
        获取 x、y 两轴的速度。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            (2,) 的速度数组 [vx, vy]。
        """
        info = next(iter(infos.values()))
        vx = float(info.get("x_velocity", 0.0))
        vy = float(info.get("y_velocity", 0.0))
        return np.array([vx, vy])

    def _get_torso_upright(self) -> float:
        """
        获取躯干的竖直程度。

        通过四元数计算躯干 z 轴在世界坐标系中的投影，
        值为 1 表示完全竖直，0 表示水平，-1 表示倒立。

        返回:
            [-1, 1] 区间内的竖直程度值。
        """
        # root 自由关节: qpos[3:7] 为四元数 (w, x, y, z)
        quat = self.single_agent_env.unwrapped.data.qpos[3:7]
        # 四元数旋转 z 轴单位向量后取 z 分量
        # R(q) * [0,0,1] 的 z 分量 = 1 - 2*(qx^2 + qy^2)
        w, qx, qy, qz = quat
        upright = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
        return float(upright)

    def _has_fallen(self) -> bool:
        """
        判断是否已跌倒。

        躯干高度低于阈值时返回 True，触发早期终止，
        避免倒地后浪费 episode 剩余步数。

        返回:
            True 表示应提前终止 episode。
        """
        return (
            self._get_body_z("torso")
            < _COMMON.min_torso_z
        )

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
        if task == "run_fwd":
            return self._run_fwd_reward(infos)
        elif task == "stand":
            return self._stand_reward(infos)
        elif task == "walk":
            return self._walk_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _standing_reward(self) -> float:
        """
        站立基底子奖励（各任务共用）。

        综合 head 高度和躯干竖直两个信号:
            - standing: head >= 目标高度时满分，低于时
                gaussian 衰减
            - upright: 躯干竖直时满分，倾斜时 linear
                衰减至 0

        返回:
            [0, 1] 区间内的奖励值。
        """
        standing = tolerance(
            self._get_geom_z("head"),
            bounds=(
                _COMMON.head_target_z,
                float("inf"),
            ),
            margin=_COMMON.head_z_margin,
        )
        upright = tolerance(
            self._get_torso_upright(),
            bounds=(0.9, float("inf")),
            margin=1.9,
            sigmoid="linear",
            value_at_margin=0,
        )
        return standing * upright

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向跑奖励。

        综合三个子奖励:
            - standing: head 高度 × 竖直（共用基底）
            - small_control: 动作平滑惩罚
            - move: 沿 x 正方向达到目标速度，
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
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        # 压缩到 [1/6, 1]，速度为零时仍保留站立激励
        move = (5 * move + 1) / 6

        return (
            self._small_control_reward()
            * self._standing_reward()
            * move
        )

    def _small_control_reward(self) -> float:
        """
        动作平滑子奖励。

        对每个执行器的控制信号用 quadratic sigmoid 衰减，
        取均值后压缩到 [0.8, 1.0] 区间，避免主导奖励
        但能有效抑制高频震颤。

        返回:
            [0.8, 1.0] 区间内的奖励值。
        """
        ctrl = self._get_control()
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

    def _stand_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        站立奖励。

        综合四个子奖励:
            - standing: head 高度达到目标（gaussian 衰减）
            - upright: 躯干保持竖直
            - small_control: 动作平滑惩罚，抑制震颤
            - dont_move: xy 两轴速度接近零

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        # xy 两轴速度约束: gaussian 衰减
        xy_vel = self._get_xy_velocity(infos)
        dont_move = float(
            tolerance(
                xy_vel,
                margin=_STAND.velocity_margin,
            ).mean()
        )

        return (
            self._small_control_reward()
            * self._standing_reward()
            * dont_move
        )

    def _walk_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        行走奖励。

        综合三个子奖励:
            - standing: head 高度 × 竖直（共用基底）
            - small_control: 动作平滑惩罚
            - move: 沿 x 正方向达到行走目标速度，
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
        move = (5 * move + 1) / 6

        return (
            self._small_control_reward()
            * self._standing_reward()
            * move
        )
