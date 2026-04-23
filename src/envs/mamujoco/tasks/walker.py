"""
Walker2d 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供 stand、walk、run 三种任务的自定义奖励函数，
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
class CommonConfig:
    """各任务共用参数。"""

    # 站立时躯干的最小目标高度（米）
    stand_height: float = 1.0
    # 站立时躯干的最大目标高度（米）
    stand_height_upper: float = 1.5
    # 直立角度容许上界（±弧度），
    # 在此范围内视为直立
    upright_angle_bound: float = float(np.deg2rad(15))
    # 健康角度下界（弧度），超出后终止 episode
    healthy_angle_low: float = float(np.deg2rad(-30))
    # 健康角度上界（弧度），超出后终止 episode
    healthy_angle_high: float = float(np.deg2rad(30))


@dataclass(frozen=True)
class StandConfig:
    """站立任务参数。"""

    # 动作惩罚 margin
    control_margin: float = 1.0


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 目标行走速度（m/s）
    speed: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""

    # 目标奔跑速度（m/s）
    speed: float = 5.0


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

    Walker2d 是一个二维平面的双足机器人，由 torso、左右大腿、
    小腿和脚组成，共 6 个受控关节。智能体分为两组（2x3），
    分别控制左右腿的三个关节。

    支持的任务集:
        - stand: 站立保持平衡
        - walk: 向前稳定行走
        - run: 向前快速奔跑
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
            angle_deg = float(
                np.rad2deg(self._get_torso_angle())
            )
            ctrl = self._get_control_magnitude()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"height={height:.2f}  "
                f"angle={angle_deg:+6.1f}°  "
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
        获取 torso 的绝对高度（qpos[1]）。

        Walker2d 的 qpos[1] 对应 rootz 关节，
        即躯干质心的 z 坐标。

        返回:
            躯干高度（米）。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[1]
        )

    def _get_torso_angle(self) -> float:
        """
        获取躯干俯仰角（qpos[2]，rooty 关节）。

        Walker2d 中 rooty 关节控制躯干绕 y 轴的旋转，
        0 表示竖直站立，正值表示前倾，负值表示后倾。

        返回:
            躯干俯仰角（弧度）。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[2]
        )

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

    def _get_control_magnitude(self) -> float:
        """
        获取当前动作信号的平均绝对值。

        返回:
            控制信号的均值幅度。
        """
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        return float(np.mean(np.abs(ctrl)))

    # ------------------------------------------------------------------
    # 奖励子函数
    # ------------------------------------------------------------------

    def _height_reward(self) -> float:
        """
        躯干高度奖励。

        躯干高度在目标范围 [1.0, 1.5] 米内时返回 1，
        低于下界时以 gaussian 方式衰减。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return tolerance(
            self._get_torso_height(),
            bounds=(
                _COMMON.stand_height,
                _COMMON.stand_height_upper,
            ),
        )

    def _upright_reward(self) -> float:
        """
        躯干直立姿态奖励。

        俯仰角在 ±15° 以内时返回 1，超出后以 gaussian
        方式衰减。margin 设为上界值，使衰减更平滑。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return tolerance(
            self._get_torso_angle(),
            bounds=(
                -_COMMON.upright_angle_bound,
                _COMMON.upright_angle_bound,
            ),
            margin=_COMMON.upright_angle_bound,
        )

    def _gait_symmetry_reward(self) -> float:
        """
        步态对称性奖励。

        通过比较左右腿对应关节角度的差异来鼓励交替步态，
        而非两腿同步运动。理想的行走步态中，左右腿应有
        约 180° 的相位差。

        利用左右大腿关节角度之差来度量:
        差值越大（交替摆动）奖励越高，
        差值为零（同步运动/跳跃）奖励为零。

        返回:
            [0, 1] 区间内的奖励值。
        """
        qpos = self.single_agent_env.unwrapped.data.qpos
        # qpos[3]: 右大腿关节角度
        # qpos[6]: 左大腿关节角度
        thigh_diff = abs(float(qpos[3] - qpos[6]))
        # 大腿关节范围 [-2.618, 0]，最大差值约 2.618
        # 差值超过 0.3 弧度（约 17°）即视为交替步态
        return tolerance(
            thigh_diff,
            bounds=(0.3, float("inf")),
            margin=0.3,
            value_at_margin=0,
            sigmoid="linear",
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
            return self._stand_reward()
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

    def _stand_reward(self) -> float:
        """
        站立奖励。

        综合三个子奖励:
            - height: 躯干高度保持在目标范围内
            - upright: 躯干保持直立姿态
            - small_control: 动作平滑惩罚，鼓励最小化
                不必要的关节运动

        返回:
            [0, 1] 区间内的奖励值。
        """
        height = self._height_reward()
        upright = self._upright_reward()

        ctrl = self.single_agent_env.unwrapped.data.ctrl
        small_control = float(
            tolerance(
                ctrl,
                margin=_STAND.control_margin,
                value_at_margin=0,
                sigmoid="quadratic",
            ).mean()
        )
        # 压缩至 [0.8, 1.0]，避免控制惩罚主导奖励
        small_control = (4 + small_control) / 5

        return height * upright * small_control

    def _walk_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        行走奖励。

        综合四个子奖励:
            - speed: 沿 x 正方向达到目标速度 1.0 m/s
            - height: 躯干高度保持在目标范围内
            - upright: 躯干保持直立姿态
            - gait: 步态对称性奖励，鼓励交替步态

        乘法组合确保智能体必须同时满足所有约束:
        仅跑快但姿态不稳不会获得高分，
        仅站稳但不前进也不会获得高分。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        speed = tolerance(
            vx,
            bounds=(_WALK.speed, float("inf")),
            margin=_WALK.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        height = self._height_reward()
        upright = self._upright_reward()
        gait = self._gait_symmetry_reward()

        # 步态奖励压缩至 [0.7, 1.0]，
        # 作为辅助信号不主导奖励
        gait = (7 + 3 * gait) / 10

        return speed * height * upright * gait

    def _run_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        奔跑奖励。

        综合三个子奖励:
            - speed: 沿 x 正方向达到目标速度 5.0 m/s
            - height: 躯干高度保持在目标范围内
            - upright: 躯干保持直立姿态

        奔跑任务不强加步态对称性约束，允许智能体自由
        探索高效的奔跑步态（包括跳跃式奔跑）。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        speed = tolerance(
            vx,
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        height = self._height_reward()
        upright = self._upright_reward()

        return speed * height * upright
