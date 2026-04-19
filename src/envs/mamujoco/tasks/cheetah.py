"""
HalfCheetah 多任务多智能体环境模块。

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
class RunConfig:
    """双足奔跑任务参数。"""

    # 目标速度（m/s）
    speed: float = 8.0
    # 躯干俯仰角容许上界（±），角度在此范围内视为直立
    upright_pitch_bound: float = float(np.deg2rad(20))


@dataclass(frozen=True)
class PostureConfig:
    """单足姿态任务的共用参数。"""

    # 姿态失败判定: 反方向翻倒阈值
    fail_pitch_low: float = float(np.deg2rad(-10))
    # 姿态失败判定: 过度倾斜阈值
    fail_pitch_high: float = float(np.deg2rad(100))
    # 支撑脚离地高度上界（米），超过则视为离地
    support_foot_z_bound: float = 0.5
    # 抬起脚的目标高度（米），达到此高度即获得满分
    raised_foot_target_z: float = 1.0
    # 脚的初始离地高度（米），用于计算抬脚奖励的 margin
    foot_initial_z: float = 0.2


@dataclass(frozen=True)
class OneFootStandConfig:
    """单足站立任务参数。"""

    # stand_bfoot 俯仰角目标区间（后肢着地）
    bfoot_pitch_low: float = float(np.deg2rad(70))
    bfoot_pitch_high: float = float(np.deg2rad(90))
    # stand_ffoot 俯仰角目标区间（前肢着地，降低以避免 head 触地）
    ffoot_pitch_low: float = float(np.deg2rad(40))
    ffoot_pitch_high: float = float(np.deg2rad(60))
    # 允许的最大水平速度（m/s），超出则 slow 奖励衰减
    max_speed: float = 1.0


@dataclass(frozen=True)
class OneFootRunConfig:
    """单足跑步任务参数。"""

    # 俯仰角目标区间下界
    pitch_low: float = float(np.deg2rad(50))
    # 俯仰角目标区间上界
    pitch_high: float = float(np.deg2rad(70))
    # 目标速度（m/s）
    speed: float = 4.0


# 全局默认配置实例
_RUN = RunConfig()
_POSTURE = PostureConfig()
_ONE_FOOT_STAND = OneFootStandConfig()
_ONE_FOOT_RUN = OneFootRunConfig()


class HalfCheetahMultiTask(MultiAgentMujocoEnv):
    """
    HalfCheetah 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 HalfCheetah，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - run_fwd: 正向跑
        - run_bwd: 反向跑
        - stand_ffoot: 前肢着地、后肢抬高姿态站立（无速度要求）
        - stand_bfoot: 后肢着地、前肢抬高姿态站立（无速度要求）
        - run_fwd_ffoot: 前肢着地、后肢抬高姿态正向跑
        - run_fwd_bfoot: 后肢着地、前肢抬高姿态正向跑
    """

    TASKS: list[str] = [
        "run_fwd",
        "run_bwd",
        "stand_ffoot",
        "stand_bfoot",
        "run_fwd_ffoot",
        "run_fwd_bfoot",
    ]

    # 需要早期终止的姿态任务集合
    _POSTURE_TASKS: frozenset[str] = frozenset({
        "stand_ffoot",
        "stand_bfoot",
        "run_fwd_ffoot",
        "run_fwd_bfoot",
    })

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 HalfCheetah 多任务环境。

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
            scenario="HalfCheetah",
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
            pitch_deg = float(np.rad2deg(self._get_torso_pitch()))
            torso_z = self._get_body_z("torso")
            bfoot_z = self._get_body_z("bfoot")
            ffoot_z = self._get_body_z("ffoot")
            print(
                f"\rtask={self.task:<15} v_x={vx:+6.2f}  "
                f"pitch={pitch_deg:+6.1f}°  "
                f"torso={torso_z:.2f}  "
                f"bfoot={bfoot_z:.2f}  "
                f"ffoot={ffoot_z:.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        # 姿态类任务的早期终止: 一旦躯干翻出可恢复范围，立即结束
        # episode，避免长 0 奖励尾巴把梯度稀释掉。
        if self._fell_out_of_posture():
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
            body_name: 刚体名称，如 "torso"、"ffoot"、
                "bfoot"。

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

    def _get_torso_pitch(self) -> float:
        """
        获取躯干俯仰角（rooty 关节位置）。

        返回:
            躯干俯仰角（弧度），0 表示水平直立，
            ±π 表示完全翻转。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[2]
        )

    def _fell_out_of_posture(self) -> bool:
        """
        判断躯干是否已倒出可恢复范围。

        仅对 _POSTURE_TASKS 中的任务生效。signed_pitch 定义与
        _one_foot_pitch_reward 一致（基于 MuJoCo rooty 实际符号）:
            stand_bfoot → pitch_sign = -1
            stand_ffoot → pitch_sign = +1
        判据:
            signed_pitch < -10° → 反方向翻倒，不可恢复
            signed_pitch > 100° → 过度倾斜，不可恢复

        返回:
            True 表示应提前终止 episode。
        """
        task = self.task
        if task not in self._POSTURE_TASKS:
            return False
        pitch_sign = (
            -1.0 if task in ("stand_bfoot", "run_fwd_bfoot")
            else 1.0
        )
        signed = pitch_sign * self._get_torso_pitch()
        return (
            signed < _POSTURE.fail_pitch_low
            or signed > _POSTURE.fail_pitch_high
        )

    def _upright_reward(self) -> float:
        """
        躯干直立姿态奖励。

        俯仰角在 ±20° 以内时返回 1，超出后立即返回 0；
        用作跑步类任务的姿态约束因子。

        返回:
            0 或 1。
        """
        return tolerance(
            self._get_torso_pitch(),
            bounds=(
                -_RUN.upright_pitch_bound,
                _RUN.upright_pitch_bound,
            ),
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
        elif task == "run_bwd":
            return self._run_bwd_reward(infos)
        elif task == "stand_ffoot":
            return self._stand_ffoot_reward(infos)
        elif task == "stand_bfoot":
            return self._stand_bfoot_reward(infos)
        elif task == "run_fwd_ffoot":
            return self._run_fwd_ffoot_reward(infos)
        elif task == "run_fwd_bfoot":
            return self._run_fwd_bfoot_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向跑奖励。

        鼓励智能体沿 x 正方向达到目标速度，同时要求
        躯干保持直立姿态，避免翻倒后靠甩动前进。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        speed = self._get_x_velocity(infos)
        speed_reward = tolerance(
            speed,
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        return speed_reward * self._upright_reward()

    def _run_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向跑奖励。

        鼓励智能体沿 x 负方向达到目标速度，同时要求
        躯干保持直立姿态，避免翻倒后靠甩动后退。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        speed = self._get_x_velocity(infos)
        speed_reward = tolerance(
            speed,
            bounds=(-float("inf"), -_RUN.speed),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        return speed_reward * self._upright_reward()

    def _one_foot_pitch_reward(
        self,
        raised_foot: str,
        pitch_low: float,
        pitch_high: float,
    ) -> float:
        """
        单足站立时的躯干倾斜子奖励。

        MuJoCo `rooty` 符号: pitch < 0 → head 朝上，
        pitch > 0 → head 朝下。

        以 signed_pitch = pitch_sign * pitch 为变量:
            [pitch_low, pitch_high] → 1（目标区间）
            < pitch_low             → reciprocal 衰减至 0
            > pitch_high            → 0（硬切断）

        参数:
            raised_foot: 需要抬起的足部名称
                ffoot 抬起 → bfoot 撑地，期望 pitch < 0
                    → pitch_sign = -1
                bfoot 抬起 → ffoot 撑地，期望 pitch > 0
                    → pitch_sign = +1
            pitch_low: 俯仰角目标区间下界（弧度）。
            pitch_high: 俯仰角目标区间上界（弧度）。

        返回:
            [0, 1] 区间内的奖励值。
        """
        pitch_sign = -1.0 if raised_foot == "ffoot" else 1.0
        signed_pitch = pitch_sign * self._get_torso_pitch()
        lower = tolerance(
            signed_pitch,
            bounds=(pitch_low, float("inf")),
            margin=pitch_low,
        )
        upper = tolerance(
            signed_pitch,
            bounds=(-float("inf"), pitch_high),
        )
        return lower * upper

    def _raised_foot_reward(
        self,
        raised_foot: str,
    ) -> float:
        """
        抬起脚的高度子奖励。

        鼓励 robot 将指定脚抬到目标高度，提供密集梯度信号。

        参数:
            raised_foot: 需要抬起的足部名称。

        返回:
            [0, 1] 区间内的奖励值。
        """
        foot_z = self._get_body_z(raised_foot)
        return tolerance(
            foot_z,
            bounds=(_POSTURE.raised_foot_target_z, float("inf")),
            margin=(
                _POSTURE.raised_foot_target_z
                - _POSTURE.foot_initial_z
            ),
            sigmoid="linear",
            value_at_margin=0,
        )

    def _stand_in_posture_reward(
        self,
        raised_foot: str,
        pitch_low: float,
        pitch_high: float,
        infos: dict[str, dict],
    ) -> float:
        """
        单足姿态站立的奖励。

        综合四个子奖励:
            - pitch: 躯干倾斜到目标角度
            - grounded: 支撑脚贴近地面
            - foot_up: 抬起脚达到目标高度
            - slow: 水平速度接近零，鼓励静止站立

        参数:
            raised_foot: 需要抬起的足部名称。
            pitch_low: 俯仰角目标区间下界（弧度）。
            pitch_high: 俯仰角目标区间上界（弧度）。
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        # raised_foot 抬起 → 反向那只脚是支撑脚
        support_foot = (
            "bfoot" if raised_foot == "ffoot" else "ffoot"
        )
        support_z = self._get_body_z(support_foot)

        # 支撑脚必须靠近地面（< 0.5m），否则给 0
        grounded = tolerance(
            support_z,
            bounds=(0.0, _POSTURE.support_foot_z_bound),
        )

        pitch = self._one_foot_pitch_reward(
            raised_foot, pitch_low, pitch_high,
        )
        foot_up = self._raised_foot_reward(raised_foot)

        # 鼓励静止: 速度在 ±max_speed 内满分，超出后线性衰减至 0
        vx = self._get_x_velocity(infos)
        slow = tolerance(
            vx,
            bounds=(
                -_ONE_FOOT_STAND.max_speed,
                _ONE_FOOT_STAND.max_speed,
            ),
            sigmoid="linear",
            value_at_margin=0,
        )

        return pitch * foot_up * grounded * slow

    def _stand_ffoot_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        前肢着地、后肢抬高姿态站立奖励。

        使用较低的 pitch 目标（40°–60°），避免躯干过度
        前倾导致 head 触地。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return self._stand_in_posture_reward(
            "bfoot",
            _ONE_FOOT_STAND.ffoot_pitch_low,
            _ONE_FOOT_STAND.ffoot_pitch_high,
            infos,
        )

    def _stand_bfoot_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        后肢着地、前肢抬高姿态站立奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return self._stand_in_posture_reward(
            "ffoot",
            _ONE_FOOT_STAND.bfoot_pitch_low,
            _ONE_FOOT_STAND.bfoot_pitch_high,
            infos,
        )

    def _run_fwd_in_posture_reward(
        self,
        raised_foot: str,
        infos: dict[str, dict],
    ) -> float:
        """
        单足姿态正向跑的奖励。

        综合四个子奖励:
            - pitch: 躯干倾斜到目标角度
            - grounded: 支撑脚贴近地面
            - foot_up: 抬起脚达到目标高度
            - speed: 沿 x 正方向达到目标速度

        参数:
            raised_foot: 需要抬起的足部名称。
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        support_foot = (
            "bfoot" if raised_foot == "ffoot" else "ffoot"
        )
        support_z = self._get_body_z(support_foot)

        grounded = tolerance(
            support_z,
            bounds=(0.0, _POSTURE.support_foot_z_bound),
        )

        pitch = self._one_foot_pitch_reward(
            raised_foot,
            _ONE_FOOT_RUN.pitch_low,
            _ONE_FOOT_RUN.pitch_high,
        )
        foot_up = self._raised_foot_reward(raised_foot)

        speed = self._get_x_velocity(infos)
        speed_reward = tolerance(
            speed,
            bounds=(_ONE_FOOT_RUN.speed, float("inf")),
            margin=_ONE_FOOT_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )

        return pitch * foot_up * grounded * speed_reward

    def _run_fwd_ffoot_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        前肢着地、后肢抬高姿态正向跑奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return self._run_fwd_in_posture_reward("bfoot", infos)

    def _run_fwd_bfoot_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        后肢着地、前肢抬高姿态正向跑奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return self._run_fwd_in_posture_reward("ffoot", infos)
