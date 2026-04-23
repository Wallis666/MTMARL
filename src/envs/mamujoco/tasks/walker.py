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
class PostureConfig:
    """姿态奖励参数。"""

    # 站立时 torso 相对于最低足部的目标高度下界（米）
    # Walker2d 初始 torso z ≈ 1.25，foot z ≈ 0，高度差 ≈ 1.25
    stand_height: float = 1.0
    # 站立高度上界（米）
    stand_height_upper: float = 2.0
    # 高度奖励 margin（米）
    height_margin: float = 0.4
    # 直立角度下界（弧度），rooty 在此范围内视为直立
    # ±15° ≈ ±0.26 rad
    upright_bound: float = float(np.deg2rad(15))
    # 直立角度 margin（弧度）
    upright_margin: float = float(np.deg2rad(30))
    # 摔倒判定: torso 高度下界（米）
    fall_height: float = 0.5
    # 摔倒判定: 俯仰角绝对值上界（弧度）
    # 60° ≈ 1.05 rad
    fall_pitch: float = float(np.deg2rad(60))


@dataclass(frozen=True)
class GaitConfig:
    """步态协调参数。"""

    # 双腿相位差奖励: 大腿关节速度符号相反时满分
    # 相位奖励的 margin 参数
    phase_margin: float = 1.0
    # 足部高度差异上界（米），双脚交替抬起的目标
    foot_diff_margin: float = 0.1
    # 步态协调奖励的权重（相对于总奖励）
    gait_weight: float = 0.1


@dataclass(frozen=True)
class EnergyConfig:
    """能量惩罚参数。"""

    # 控制量惩罚 margin
    control_margin: float = 1.0
    # 能量奖励权重
    energy_weight: float = 0.1


@dataclass(frozen=True)
class StandConfig:
    """站立任务参数。"""

    # 静止速度 margin（m/s），速度越小奖励越高
    speed_margin: float = 0.5
    # 双脚 x 方向间距目标下界（米）
    foot_gap_lower: float = 0.15
    # 双脚 x 方向间距目标上界（米）
    foot_gap_upper: float = 0.3
    # 间距 margin（米）
    foot_gap_margin: float = 0.15


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 前进目标速度（m/s）
    fwd_speed: float = 1.0
    # 后退目标速度（m/s），用绝对值表示
    bwd_speed: float = 1.0
    # 速度 margin
    speed_margin: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""

    # 前进目标速度（m/s）
    fwd_speed: float = 4.0
    # 后退目标速度（m/s），用绝对值表示
    bwd_speed: float = 4.0
    # 速度 margin
    speed_margin: float = 3.0


# 全局默认配置实例
_POSTURE = PostureConfig()
_GAIT = GaitConfig()
_ENERGY = EnergyConfig()
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
        - stand: 站立保持平衡（速度接近零）
        - walk_fwd: 向前行走（目标速度 1.0 m/s）
        - walk_bwd: 向后行走（目标速度 -1.0 m/s）
        - run_fwd: 向前奔跑（目标速度 4.0 m/s）
        - run_bwd: 向后奔跑（目标速度 -4.0 m/s）

    奖励设计要点:
        1. 姿态奖励作为乘性基础项，确保机器人始终
           优先维持站立平衡
        2. 步态协调奖励鼓励双腿交替运动，避免双腿
           同步导致的蹦跳步态
        3. 能量奖励惩罚过大的控制信号，促进平滑动作
        4. 速度奖励根据任务类型追踪不同的目标速度
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
            height = self._get_height()
            pitch_deg = float(
                np.rad2deg(self._get_torso_pitch())
            )
            ctrl = self._get_control_magnitude()
            gait = self._gait_phase_reward()
            foot_gap = self._get_foot_gap()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"height={height:.2f}  "
                f"pitch={pitch_deg:+6.1f}°  "
                f"ctrl={ctrl:.2f}  "
                f"gait={gait:.2f}  "
                f"gap={foot_gap:.3f}  "
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

    def _get_body_z(
        self,
        body_name: str,
    ) -> float:
        """
        获取指定刚体的 z 轴高度。

        参数:
            body_name: 刚体名称，如 "torso"、"foot"。

        返回:
            该刚体当前的 z 坐标值。
        """
        return float(
            self.single_agent_env.unwrapped.data.body(
                body_name
            ).xpos[2]
        )

    def _get_height(self) -> float:
        """
        获取 torso 相对于最低足部的高度差。

        取左右脚中较低的一只作为参考，避免单脚
        抬起时高度计算失真。

        返回:
            torso 与最低足部的 z 坐标差值（米）。
        """
        foot_z = min(
            self._get_body_z("foot"),
            self._get_body_z("foot_left"),
        )
        return self._get_body_z("torso") - foot_z

    def _get_torso_pitch(self) -> float:
        """
        获取躯干俯仰角（rooty 关节位置）。

        Walker2d 中 rooty 位于 qpos[2]，正值表示前倾，
        负值表示后仰。

        返回:
            躯干俯仰角（弧度）。
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

    def _get_thigh_velocities(self) -> tuple[float, float]:
        """
        获取左右大腿关节的角速度。

        Walker2d qvel 布局:
            [0]=rootx, [1]=rootz, [2]=rooty,
            [3]=thigh_joint, [4]=leg_joint, [5]=foot_joint,
            [6]=thigh_left_joint, [7]=leg_left_joint,
            [8]=foot_left_joint

        返回:
            (右大腿角速度, 左大腿角速度) 元组。
        """
        qvel = self.single_agent_env.unwrapped.data.qvel
        return float(qvel[3]), float(qvel[6])

    def _get_foot_heights(self) -> tuple[float, float]:
        """
        获取左右脚的 z 轴高度。

        返回:
            (右脚高度, 左脚高度) 元组。
        """
        return (
            self._get_body_z("foot"),
            self._get_body_z("foot_left"),
        )

    def _get_foot_gap(self) -> float:
        """
        获取左右脚在 x 方向的绝对间距。

        返回:
            双脚 x 坐标差的绝对值（米）。
        """
        data = self.single_agent_env.unwrapped.data
        foot_r_x = float(data.body("foot").xpos[0])
        foot_l_x = float(
            data.body("foot_left").xpos[0]
        )
        return abs(foot_r_x - foot_l_x)

    # ------------------------------------------------------------------
    # 早期终止
    # ------------------------------------------------------------------

    def _has_fallen(self) -> bool:
        """
        判断是否应提前终止 episode。

        满足以下任一条件时终止:
            - 状态包含 NaN 或 Inf
            - torso 高度差 < fall_height
            - 俯仰角绝对值 > fall_pitch

        返回:
            True 表示应提前终止。
        """
        qpos = self.single_agent_env.unwrapped.data.qpos
        qvel = self.single_agent_env.unwrapped.data.qvel
        # NaN / Inf 检测
        if (
            np.any(np.isnan(qpos))
            or np.any(np.isinf(qpos))
            or np.any(np.isnan(qvel))
            or np.any(np.isinf(qvel))
        ):
            return True
        # 高度过低
        if self._get_height() < _POSTURE.fall_height:
            return True
        # 俯仰角过大
        if abs(self._get_torso_pitch()) > _POSTURE.fall_pitch:
            return True
        return False

    # ------------------------------------------------------------------
    # 子奖励：姿态
    # ------------------------------------------------------------------

    def _posture_reward(self) -> float:
        """
        姿态子奖励。

        综合两个因子:
            - standing: torso 保持在目标高度范围内
            - upright: 俯仰角保持在直立范围内

        两者相乘，确保机器人必须同时满足高度和直立
        要求才能获得高奖励。

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 高度奖励: 在目标范围内满分，低于时衰减
        standing = tolerance(
            self._get_height(),
            bounds=(
                _POSTURE.stand_height,
                _POSTURE.stand_height_upper,
            ),
            margin=_POSTURE.height_margin,
        )
        # 直立奖励: 俯仰角在 ±15° 内满分
        pitch = abs(self._get_torso_pitch())
        upright = tolerance(
            pitch,
            bounds=(0.0, _POSTURE.upright_bound),
            margin=_POSTURE.upright_margin,
            sigmoid="linear",
        )
        return standing * upright

    # ------------------------------------------------------------------
    # 子奖励：步态协调
    # ------------------------------------------------------------------

    def _gait_phase_reward(self) -> float:
        """
        步态协调子奖励。

        通过检测左右大腿关节角速度的反相关程度来
        奖励交替步态。当两腿角速度符号相反且幅度
        相当时，表明双腿在做交替摆动（行走步态）。

        计算方式:
            phase_diff = -v_right × v_left
            当 phase_diff > 0（符号相反）时给予奖励。

        返回:
            [0, 1] 区间内的奖励值。
        """
        v_right, v_left = self._get_thigh_velocities()
        # 负积 > 0 表示交替运动
        phase_product = -v_right * v_left
        return float(
            tolerance(
                phase_product,
                bounds=(0.0, float("inf")),
                margin=_GAIT.phase_margin,
                sigmoid="linear",
                value_at_margin=0.1,
            )
        )

    # ------------------------------------------------------------------
    # 子奖励：能量
    # ------------------------------------------------------------------

    def _energy_reward(self) -> float:
        """
        能量子奖励。

        惩罚过大的控制信号，鼓励平滑动作。
        使用 tolerance 对控制量均值进行 quadratic 衰减，
        然后映射到 [0.8, 1.0] 范围，避免过度惩罚。

        返回:
            [0.8, 1.0] 区间内的奖励值。
        """
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        small_control = float(
            tolerance(
                ctrl,
                margin=_ENERGY.control_margin,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
        )
        # 映射到 [0.8, 1.0]
        return (4 + small_control) / 5

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
            return self._walk_reward(infos, forward=True)
        elif task == "walk_bwd":
            return self._walk_reward(infos, forward=False)
        elif task == "run_fwd":
            return self._run_reward(infos, forward=True)
        elif task == "run_bwd":
            return self._run_reward(infos, forward=False)
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

        保持姿态直立、速度接近零、控制量小，
        且双脚保持适当间距（不重叠）。
        总奖励 = 姿态 × 速度 × 能量 × 脚间距。

        站立任务不加入步态协调奖励，因为站立时
        双腿应保持静止而非交替运动。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 速度接近零时满分
        vx = self._get_x_velocity(infos)
        speed_reward = tolerance(
            vx,
            bounds=(0.0, 0.0),
            margin=_STAND.speed_margin,
            value_at_margin=0.01,
        )
        # 双脚间距在目标范围内时满分
        foot_gap_reward = tolerance(
            self._get_foot_gap(),
            bounds=(
                _STAND.foot_gap_lower,
                _STAND.foot_gap_upper,
            ),
            margin=_STAND.foot_gap_margin,
        )
        return (
            self._posture_reward()
            * speed_reward
            * self._energy_reward()
            * foot_gap_reward
        )

    def _walk_reward(
        self,
        infos: dict[str, dict],
        forward: bool,
    ) -> float:
        """
        行走任务奖励。

        保持姿态直立，沿指定方向达到目标步行速度，
        并鼓励交替步态。

        奖励构成:
            主项 = 姿态 × 速度 × 能量
            步态 = 步态协调奖励
            总奖励 = 主项 × (1 - gait_weight)
                     + 主项 × 步态 × gait_weight

        步态奖励以加权形式加入，而非直接相乘，
        避免训练初期因步态不协调导致奖励过低、
        阻碍策略探索。

        参数:
            infos: 环境 step 返回的信息字典。
            forward: True 表示前进，False 表示后退。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        if forward:
            speed_reward = tolerance(
                vx,
                bounds=(
                    _WALK.fwd_speed, _WALK.fwd_speed,
                ),
                margin=_WALK.speed_margin,
                sigmoid="linear",
            )
        else:
            speed_reward = tolerance(
                -vx,
                bounds=(
                    _WALK.bwd_speed, _WALK.bwd_speed,
                ),
                margin=_WALK.speed_margin,
                sigmoid="linear",
            )
        main = (
            self._posture_reward()
            * speed_reward
            * self._energy_reward()
        )
        gait = self._gait_phase_reward()
        w = _GAIT.gait_weight
        return main * (1 - w) + main * gait * w

    def _run_reward(
        self,
        infos: dict[str, dict],
        forward: bool,
    ) -> float:
        """
        奔跑任务奖励。

        保持姿态直立，沿指定方向达到目标奔跑速度。
        速度达到目标值即满分，更快不扣分。

        奖励构成与行走类似，但步态协调的权重略低，
        因为高速奔跑时可能出现飞行相（双脚同时
        离地），此时步态相位检测不完全适用。

        参数:
            infos: 环境 step 返回的信息字典。
            forward: True 表示前进，False 表示后退。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        if forward:
            speed_reward = tolerance(
                vx,
                bounds=(_RUN.fwd_speed, float("inf")),
                margin=_RUN.speed_margin,
                value_at_margin=0,
                sigmoid="linear",
            )
        else:
            speed_reward = tolerance(
                -vx,
                bounds=(_RUN.bwd_speed, float("inf")),
                margin=_RUN.speed_margin,
                value_at_margin=0,
                sigmoid="linear",
            )
        main = (
            self._posture_reward()
            * speed_reward
            * self._energy_reward()
        )
        gait = self._gait_phase_reward()
        # 奔跑时步态权重减半，允许飞行相
        w = _GAIT.gait_weight / 2
        return main * (1 - w) + main * gait * w
