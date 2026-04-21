"""
Walker2d 多任务多智能体环境模块。

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
class StandConfig:
    """站立任务参数。"""

    # torso 最低高度（米），高于此值视为站立
    stand_height: float = 1.2


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # torso 最低高度（米）
    stand_height: float = 1.2
    # 目标速度（m/s）
    speed: float = 2.0


@dataclass(frozen=True)
class RunConfig:
    """跑步任务参数。"""

    # torso 最低高度（米）
    stand_height: float = 1.2
    # 目标速度（m/s）
    speed: float = 8.0


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
        - walk_fwd: 正向行走
        - walk_bwd: 反向行走
        - run_fwd: 正向跑步
        - run_bwd: 反向跑步
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
            torso_z = self._get_body_z("torso")
            upright = self._get_torso_upright()
            pitch_deg = float(
                np.rad2deg(self._get_torso_pitch())
            )
            gait = self._gait_reward()
            print(
                f"\rtask={self.task:<12} "
                f"v_x={vx:+6.2f}  "
                f"torso={torso_z:.2f}  "
                f"upright={upright:+.2f}  "
                f"pitch={pitch_deg:+6.1f}°  "
                f"gait={gait:.2f}  "
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

    def _get_body_z(
        self,
        body_name: str,
    ) -> float:
        """
        获取指定刚体的 z 轴高度。

        参数:
            body_name: 刚体名称，如 "torso"。

        返回:
            该刚体当前的 z 坐标值。
        """
        return float(
            self.single_agent_env.unwrapped.data.body(
                body_name
            ).xpos[2]
        )

    def _get_torso_pitch(self) -> float:
        """
        获取躯干俯仰角（rooty 关节位置）。

        返回:
            躯干俯仰角（弧度）。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[2]
        )

    def _get_torso_upright(self) -> float:
        """
        获取躯干的竖直程度。

        通过旋转矩阵 z-z 分量计算，
        值为 1 表示完全竖直，-1 表示完全倒立。

        返回:
            [-1, 1] 区间内的竖直程度值。
        """
        # rooty 是 hinge 关节，pitch 角度即 qpos[2]
        # upright = cos(pitch)
        pitch = self._get_torso_pitch()
        return float(np.cos(pitch))

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
    # 共用子奖励
    # ------------------------------------------------------------------

    def _stand_reward_base(
        self,
        stand_height: float,
    ) -> float:
        """
        站立基底子奖励。

        综合 torso 高度和竖直程度两个信号，用加权平均
        而非乘法组合，保证部分达标时仍有梯度。

        参数:
            stand_height: 目标站立高度（米）。

        返回:
            [0, 1] 区间内的奖励值。
        """
        standing = tolerance(
            self._get_body_z("torso"),
            bounds=(stand_height, float("inf")),
            margin=stand_height / 2,
        )
        upright = (1 + self._get_torso_upright()) / 2
        return (3 * standing + upright) / 4

    def _gait_reward(self) -> float:
        """
        步态交替奖励。

        通过左右脚垂直速度的反相程度衡量交替步态：
        一脚抬起（vz > 0）同时另一脚落下（vz < 0）时
        乘积为负，取反后为正奖励。
        静态抬腿时速度为零，不会被奖励。
        压缩到 [0.5, 1]，避免完全压制其他奖励分量。

        返回:
            [0.5, 1] 区间内的奖励值。
        """
        data = self.single_agent_env.unwrapped.data
        right_vz = float(data.body("foot").cvel[5])
        left_vz = float(data.body("foot_left").cvel[5])
        # 反相时乘积为负，取反得正值；同相或静止时为零或负
        anti_phase = -right_vz * left_vz
        alternating = tolerance(
            anti_phase,
            bounds=(0.0, float("inf")),
            margin=1.0,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return (alternating + 1) / 2

    def _move_reward(
        self,
        infos: dict[str, dict],
        speed: float,
    ) -> float:
        """
        移动速度子奖励。

        speed > 0 为正向，speed < 0 为反向。

        参数:
            infos: 环境 step 返回的信息字典。
            speed: 目标速度（m/s），正值为正向，
                负值为反向。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        if speed > 0:
            return tolerance(
                vx,
                bounds=(speed, float("inf")),
                margin=speed / 2,
                value_at_margin=0.5,
                sigmoid="linear",
            )
        return tolerance(
            vx,
            bounds=(-float("inf"), speed),
            margin=abs(speed) / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _stand_reward(self) -> float:
        """
        站立奖励。

        仅要求保持站立姿态，无速度要求。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return self._stand_reward_base(_STAND.stand_height)

    def _walk_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向行走奖励。

        综合站立、速度和步态交替三个子奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return (
            self._stand_reward_base(_WALK.stand_height)
            * self._move_reward(infos, _WALK.speed)
            * self._gait_reward()
        )

    def _walk_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向行走奖励。

        综合站立、速度和步态交替三个子奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return (
            self._stand_reward_base(_WALK.stand_height)
            * self._move_reward(infos, -_WALK.speed)
            * self._gait_reward()
        )

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向跑步奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return (
            self._stand_reward_base(_RUN.stand_height)
            * self._move_reward(infos, _RUN.speed)
            * self._gait_reward()
        )

    def _run_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向跑步奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return (
            self._stand_reward_base(_RUN.stand_height)
            * self._move_reward(infos, -_RUN.speed)
            * self._gait_reward()
        )
