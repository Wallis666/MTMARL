"""
Humanoid 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供 stand、walk、run 任务的自定义奖励函数，支持在任务间
动态切换。
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
class PostureConfig:
    """姿态奖励参数。"""

    # 标准站立时的头部高度（米）
    # gymnasium-robotics Humanoid 的 head geom 初始 z ≈ 1.59
    stand_height: float = 1.59
    # 头部高度因子（× stand_height 为奖励下界）
    head_factor: float = 0.95
    # 头部高度奖励 margin（米）
    head_margin: float = 0.5
    # 躯干直立下界（up 向量的 z 分量）
    upright_bound: float = 0.9
    # 躯干直立 margin
    upright_margin: float = 0.9
    # 盆骨高度因子（× stand_height 为下界）
    pelvis_factor: float = 0.6
    # 摔倒判定: torso 高度下界（米），低于则终止
    # 与 Gymnasium 默认 healthy_z_range 下界一致
    fall_height: float = 1.0
    # 摔倒判定: 躯干直立下界（up.z），低于则终止
    # 0.7 ≈ arccos(0.7) ≈ 45.6°，倾斜超过约 46° 终止
    fall_upright: float = 0.7
    # 摔倒判定: 头部高度因子（× stand_height 为终止下界）
    # 头部低于 stand_height * 0.5 ≈ 0.80m 时终止
    head_fall_factor: float = 0.5
    # 摔倒判定: 关节速度异常阈值（rad/s 或 m/s）
    max_qvel: float = 200.0


@dataclass(frozen=True)
class GaitConfig:
    """步态奖励参数。"""

    # 朝向与目标方向点积的下界
    heading_bound: float = 0.9
    # 朝向 margin
    heading_margin: float = 0.3
    # 盆骨水平度下界（up.z 分量）
    pelvis_level_bound: float = 0.9
    # 盆骨水平度 margin
    pelvis_level_margin: float = 0.3
    # 足部离地高度上界（米）
    foot_height_bound: float = 0.3
    # 足部高度 margin（米）
    foot_height_margin: float = 0.5


@dataclass(frozen=True)
class EnergyConfig:
    """能量奖励参数。"""

    # 能量惩罚系数
    coef: float = 0.1


@dataclass(frozen=True)
class StandConfig:
    """站立任务参数。"""

    # 静止速度 margin（m/s）
    speed_margin: float = 1.0


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 目标速度（m/s）
    speed: float = 2.0


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""

    # 目标速度（m/s）
    speed: float = 8.0


# 全局默认配置实例
_POSTURE = PostureConfig()
_GAIT = GaitConfig()
_ENERGY = EnergyConfig()
_STAND = StandConfig()
_WALK = WalkConfig()
_RUN = RunConfig()


class HumanoidMultiTask(MultiAgentMujocoEnv):
    """
    Humanoid 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Humanoid，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - stand: 站立（保持姿态，速度接近零）
        - walk: 行走（保持姿态，达到步行速度）
        - run: 奔跑（保持姿态，达到奔跑速度）
    """

    TASKS: list[str] = [
        "stand",
        "walk",
        "run",
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
            head_z = self._get_geom_z("head")
            torso_z = self._get_body_z("torso")
            upright = self._get_body_up_z("torso")
            print(
                f"\rtask={self.task:<6} "
                f"v_x={vx:+6.2f}  "
                f"head={head_z:.2f}  "
                f"torso={torso_z:.2f}  "
                f"upright={upright:.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        # 摔倒时提前终止 episode
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
        info = next(iter(infos.values()))
        return float(info.get("x_velocity", 0.0))

    def _get_y_velocity(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        从信息字典中提取 y 方向速度。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            y 方向线速度。
        """
        info = next(iter(infos.values()))
        return float(info.get("y_velocity", 0.0))

    def _get_horizontal_speed(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        计算水平面速度大小。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            水平面速度大小（m/s）。
        """
        vx = self._get_x_velocity(infos)
        vy = self._get_y_velocity(infos)
        return float(np.hypot(vx, vy))

    def _get_body_z(
        self,
        body_name: str,
    ) -> float:
        """
        获取指定刚体的 z 轴高度。

        参数:
            body_name: 刚体名称。

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
            geom_name: 几何体名称。

        返回:
            该几何体当前的 z 坐标值。
        """
        env = self.single_agent_env.unwrapped
        geom_id = env.model.geom(geom_name).id
        return float(env.data.geom_xpos[geom_id][2])

    def _get_body_up_z(
        self,
        body_name: str,
    ) -> float:
        """
        获取刚体 up 向量的 z 分量（直立/水平度）。

        旋转矩阵 xmat 按行展平为 9 元素数组，
        xmat[8] 为 up 向量的 z 分量：
        1.0 表示完全竖直（直立）或完全水平（盆骨）。

        参数:
            body_name: 刚体名称。

        返回:
            [-1, 1] 区间内的值。
        """
        xmat = (
            self.single_agent_env.unwrapped.data
            .body(body_name).xmat
        )
        return float(xmat[8])

    def _get_body_heading(
        self,
        body_name: str,
    ) -> float:
        """
        获取刚体前方向与 x 正方向的对齐度。

        将前方向投影到 x-y 平面并归一化后，计算与
        目标方向 [1, 0] 的点积。

        参数:
            body_name: 刚体名称。

        返回:
            [-1, 1] 区间内的对齐度，1 为完全对齐。
        """
        xmat = (
            self.single_agent_env.unwrapped.data
            .body(body_name).xmat
        )
        fx, fy = float(xmat[0]), float(xmat[1])
        norm = float(np.hypot(fx, fy))
        if norm < 1e-8:
            return 0.0
        return fx / norm

    def _get_geom_heading(
        self,
        geom_name: str,
    ) -> float:
        """
        获取几何体前方向与 x 正方向的对齐度。

        参数:
            geom_name: 几何体名称。

        返回:
            [-1, 1] 区间内的对齐度，1 为完全对齐。
        """
        env = self.single_agent_env.unwrapped
        geom_id = env.model.geom(geom_name).id
        xmat = env.data.geom_xmat[geom_id]
        fx, fy = float(xmat[0]), float(xmat[1])
        norm = float(np.hypot(fx, fy))
        if norm < 1e-8:
            return 0.0
        return fx / norm

    def _get_max_foot_height(self) -> float:
        """
        获取双脚中较高一只的 z 轴高度。

        返回:
            双脚中较大的 z 坐标值。
        """
        return max(
            self._get_body_z("right_foot"),
            self._get_body_z("left_foot"),
        )

    def _get_ctrl(self) -> NDArray:
        """
        获取当前执行器控制量。

        gymnasium-robotics Humanoid 有 17 个执行器
        （无 ankle 关节，dm_control 版本为 21 个）。

        返回:
            17 维控制量数组。
        """
        return (
            self.single_agent_env.unwrapped.data.ctrl
            .copy()
        )

    def _get_qvel(self) -> NDArray:
        """
        获取所有关节速度。

        返回:
            23 维关节速度数组（含 root 的 6 自由度
            + 17 个铰链关节）。
        """
        return (
            self.single_agent_env.unwrapped.data.qvel
            .copy()
        )

    # ------------------------------------------------------------------
    # 早期终止
    # ------------------------------------------------------------------

    def _has_fallen(self) -> bool:
        """
        判断是否应提前终止 episode。

        满足以下任一条件时终止:
            - 状态包含 NaN 或 Inf
            - 头部高度 < stand_height × head_fall_factor
            - 躯干倾斜度 < fall_upright
            - 任意关节速度绝对值 > max_qvel

        注: torso 高度 < fall_height 由 Gymnasium 底层
        healthy_z_range=(1.0, 2.0) 处理，此处不重复检查。

        返回:
            True 表示应提前终止。
        """
        # NaN / Inf 检测
        qpos = self.single_agent_env.unwrapped.data.qpos
        qvel = self._get_qvel()
        if (
            np.any(np.isnan(qpos))
            or np.any(np.isinf(qpos))
            or np.any(np.isnan(qvel))
            or np.any(np.isinf(qvel))
        ):
            return True
        # 头部高度过低
        head_fall = (
            _POSTURE.stand_height
            * _POSTURE.head_fall_factor
        )
        if self._get_geom_z("head") < head_fall:
            return True
        # 躯干倾斜过大
        if (
            self._get_body_up_z("torso")
            < _POSTURE.fall_upright
        ):
            return True
        # 关节速度异常
        if np.any(np.abs(qvel) > _POSTURE.max_qvel):
            return True
        return False

    # ------------------------------------------------------------------
    # 子奖励：姿态
    # ------------------------------------------------------------------

    def _posture_reward(self) -> float:
        """
        姿态子奖励。

        综合三个因子:
            - standing: 头部保持在目标高度以上
            - upright: 躯干保持竖直
            - pelvis_height: 盆骨保持在合理高度

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 头部高度 ≥ stand_height * head_factor 时满分
        standing = tolerance(
            self._get_geom_z("head"),
            bounds=(
                _POSTURE.stand_height
                * _POSTURE.head_factor,
                float("inf"),
            ),
            margin=_POSTURE.head_margin,
        )
        # 躯干 up.z ≥ 0.9 时满分
        upright = tolerance(
            self._get_body_up_z("torso"),
            bounds=(_POSTURE.upright_bound, float("inf")),
            sigmoid="linear",
            margin=_POSTURE.upright_margin,
        )
        # 盆骨高度 ≥ stand_height * 0.6 时满分
        pelvis_bound = (
            _POSTURE.stand_height * _POSTURE.pelvis_factor
        )
        pelvis_height = tolerance(
            self._get_body_z("pelvis"),
            bounds=(pelvis_bound, float("inf")),
            sigmoid="linear",
            margin=pelvis_bound,
        )
        return standing * upright * pelvis_height

    # ------------------------------------------------------------------
    # 子奖励：步态
    # ------------------------------------------------------------------

    def _gait_reward(self) -> float:
        """
        步态子奖励。

        综合五个因子:
            - torso_heading: 躯干正对目标方向
            - head_heading: 头部正对目标方向
            - pelvis_yaw: 盆骨正对目标方向
            - pelvis_level: 盆骨保持水平
            - feet_height: 足部贴近地面

        返回:
            [0, 1] 区间内的奖励值。
        """
        torso_heading = tolerance(
            self._get_body_heading("torso"),
            bounds=(
                _GAIT.heading_bound, 1.0,
            ),
            margin=_GAIT.heading_margin,
            sigmoid="linear",
        )
        head_heading = tolerance(
            self._get_geom_heading("head"),
            bounds=(
                _GAIT.heading_bound, 1.0,
            ),
            margin=_GAIT.heading_margin,
            sigmoid="linear",
        )
        pelvis_yaw = tolerance(
            self._get_body_heading("pelvis"),
            bounds=(
                _GAIT.heading_bound, 1.0,
            ),
            margin=_GAIT.heading_margin,
            sigmoid="linear",
        )
        pelvis_level = tolerance(
            self._get_body_up_z("pelvis"),
            bounds=(
                _GAIT.pelvis_level_bound, 1.0,
            ),
            margin=_GAIT.pelvis_level_margin,
            sigmoid="linear",
        )
        feet_height = tolerance(
            self._get_max_foot_height(),
            bounds=(0.0, _GAIT.foot_height_bound),
            margin=_GAIT.foot_height_margin,
            sigmoid="quadratic",
        )
        return (
            torso_heading
            * head_heading
            * pelvis_yaw
            * pelvis_level
            * feet_height
        )

    # ------------------------------------------------------------------
    # 子奖励：能量
    # ------------------------------------------------------------------

    def _energy_reward(self) -> float:
        """
        能量子奖励。

        惩罚过大的控制量，鼓励节能运动:
            energy_reward = exp(-coef × mean(ctrl²))

        返回:
            (0, 1] 区间内的奖励值。
        """
        ctrl = self._get_ctrl()
        return float(
            np.exp(
                -_ENERGY.coef * np.mean(ctrl ** 2)
            )
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
        if task == "stand":
            return self._stand_reward(infos)
        elif task == "walk":
            return self._walk_reward(infos)
        elif task == "run":
            return self._run_reward(infos)
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

        保持姿态直立且速度接近零。
        总奖励 = 姿态 × 速度 × 能量 × 步态。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        speed = self._get_horizontal_speed(infos)
        speed_reward = tolerance(
            speed,
            bounds=(0.0, 0.0),
            margin=_STAND.speed_margin,
            value_at_margin=0.01,
        )
        return (
            self._posture_reward()
            * speed_reward
            * self._energy_reward()
            * self._gait_reward()
        )

    def _walk_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        行走任务奖励。

        保持姿态直立并沿 x 正方向达到目标步行速度。
        速度以精确目标值为中心，过快过慢均会衰减。
        总奖励 = 姿态 × 速度 × 能量 × 步态。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        speed_reward = tolerance(
            vx,
            bounds=(_WALK.speed, _WALK.speed),
            margin=_WALK.speed,
            sigmoid="linear",
        )
        return (
            self._posture_reward()
            * speed_reward
            * self._energy_reward()
            * self._gait_reward()
        )

    def _run_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        奔跑任务奖励。

        保持姿态直立并沿 x 正方向达到目标奔跑速度。
        速度达到目标值即满分，更快不扣分。
        总奖励 = 姿态 × 速度 × 能量 × 步态。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        speed_reward = tolerance(
            vx,
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        return (
            self._posture_reward()
            * speed_reward
            * self._energy_reward()
            * self._gait_reward()
        )
