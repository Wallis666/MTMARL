"""
Hopper 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供 stand 和 hop 两种任务的自定义奖励函数，
支持在任务间动态切换。
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

    # torso 相对于 foot 的最小高度差（米），
    # 高于此值视为站立
    stand_height: float = 0.6
    # 站立高度上界（米）
    stand_height_upper: float = 2.0
    # 动作惩罚 margin
    control_margin: float = 1.0


@dataclass(frozen=True)
class HopConfig:
    """跳跃任务参数。"""

    # torso 相对于 foot 的最小高度差（米）
    stand_height: float = 0.6
    # 站立高度上界（米）
    stand_height_upper: float = 2.0
    # 目标水平速度（m/s）
    speed: float = 4.0


# 全局默认配置实例
_STAND = StandConfig()
_HOP = HopConfig()


class HopperMultiTask(MultiAgentMujocoEnv):
    """
    Hopper 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Hopper，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - stand: 站立保持平衡
        - hop: 向前跳跃
    """

    TASKS: list[str] = [
        "stand",
        "hop",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Hopper 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "3x1" 表示
                3 个智能体各控制 1 个关节。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        super().__init__(
            scenario="Hopper",
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
            torso_z = self._get_body_z("torso")
            foot_z = self._get_body_z("foot")
            pitch_deg = float(
                np.rad2deg(self._get_torso_pitch())
            )
            ctrl = self._get_control_magnitude()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"height={height:.2f}  "
                f"torso={torso_z:.2f}  "
                f"foot={foot_z:.2f}  "
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
        获取 torso 相对于 foot 的高度差。

        返回:
            torso 与 foot 的 z 坐标差值（米）。
        """
        return (
            self._get_body_z("torso")
            - self._get_body_z("foot")
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

    def _get_control_magnitude(self) -> float:
        """
        获取当前动作信号的平均绝对值。

        返回:
            控制信号的均值幅度。
        """
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        return float(np.mean(np.abs(ctrl)))

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
        elif task == "hop":
            return self._hop_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _stand_reward(self) -> float:
        """
        站立奖励。

        综合两个子奖励:
            - standing: torso 相对 foot 高度在
                目标范围内时满分，低于时 gaussian 衰减
            - small_control: 动作平滑惩罚，quadratic
                衰减后压缩到 [0.8, 1.0]

        返回:
            [0, 1] 区间内的奖励值。
        """
        standing = tolerance(
            self._get_height(),
            bounds=(
                _STAND.stand_height,
                _STAND.stand_height_upper,
            ),
        )
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        small_control = float(
            tolerance(
                ctrl,
                margin=_STAND.control_margin,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
        )
        small_control = (4 + small_control) / 5

        return standing * small_control

    def _hop_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        跳跃奖励。

        综合两个子奖励:
            - standing: torso 相对 foot 高度在
                目标范围内时满分
            - hopping: 沿 x 正方向达到目标速度

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        standing = tolerance(
            self._get_height(),
            bounds=(
                _HOP.stand_height,
                _HOP.stand_height_upper,
            ),
        )

        vx = self._get_x_velocity(infos)
        hopping = tolerance(
            vx,
            bounds=(_HOP.speed, float("inf")),
            margin=_HOP.speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )

        return standing * hopping
