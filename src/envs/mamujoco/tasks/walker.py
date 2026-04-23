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
    # 接触力归一化阈值（N），用于判定脚是否着地
    contact_threshold: float = 1.0
    # 双脚使用率滑动窗口长度（步）
    foot_usage_window: int = 40
    # 双脚使用率的最小均衡比例，低于此值时开始惩罚
    foot_usage_min_ratio: float = 0.3


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
        # 上一步的动作，用于计算动作平滑奖励
        self._prev_actions: NDArray | None = None
        # 左右脚接触历史，用于计算双脚使用率
        self._right_contact_history: list[bool] = []
        self._left_contact_history: list[bool] = []

    # ------------------------------------------------------------------
    # 重写 reset：清空历史状态
    # ------------------------------------------------------------------

    def reset(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, NDArray], dict[str, dict]]:
        """
        重置环境并清空步态追踪历史。

        参数:
            *args: 传递给父类 reset 的位置参数。
            **kwargs: 传递给父类 reset 的关键字参数。

        返回:
            (观测, 信息) 二元组。
        """
        obs, info = super().reset(*args, **kwargs)
        self._prev_actions = None
        self._right_contact_history.clear()
        self._left_contact_history.clear()
        return obs, info

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
        # 记录当前动作用于平滑奖励
        current_ctrl = \
            self.single_agent_env.unwrapped.data.ctrl.copy()

        obs, _, terms, truncs, infos = super().step(actions)

        # 更新接触历史
        r_frc, l_frc = self._get_foot_contact_forces()
        self._right_contact_history.append(
            r_frc > _WALK.contact_threshold
        )
        self._left_contact_history.append(
            l_frc > _WALK.contact_threshold
        )
        # 保持窗口长度
        window = _WALK.foot_usage_window
        if len(self._right_contact_history) > window:
            self._right_contact_history.pop(0)
            self._left_contact_history.pop(0)

        task_reward = self._compute_reward(infos)
        rewards = {agent: task_reward for agent in obs}

        # 更新上一步动作
        self._prev_actions = current_ctrl

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
                f"r_frc={r_frc:.0f}  "
                f"l_frc={l_frc:.0f}  "
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

    def _get_foot_contact_forces(self) -> tuple[float, float]:
        """
        获取左右脚的地面接触法向力。

        遍历 MuJoCo 的接触点列表，累加每只脚与地面之间
        的法向力分量。

        返回:
            (右脚接触力, 左脚接触力) 元组，单位为牛顿。
        """
        env = self.single_agent_env.unwrapped
        data = env.data
        model = env.model
        right_foot_id = model.geom("foot_geom").id
        left_foot_id = model.geom("foot_left_geom").id
        floor_id = model.geom("floor").id

        right_force = 0.0
        left_force = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            # 提取法向力分量
            if contact.efc_address < 0:
                continue
            normal_force = abs(float(
                data.efc_force[contact.efc_address]
            ))
            if {geom1, geom2} == {floor_id, right_foot_id}:
                right_force += normal_force
            elif {geom1, geom2} == {floor_id, left_foot_id}:
                left_force += normal_force

        return right_force, left_force

    def _get_foot_velocities(self) -> tuple[float, float]:
        """
        获取左右脚在 x-z 平面上的速度大小。

        通过 MuJoCo 的刚体速度接口获取脚部质心速度，
        取 x 和 z 分量的范数作为脚的运动速度。

        返回:
            (右脚速度, 左脚速度) 元组，单位为 m/s。
        """
        env = self.single_agent_env.unwrapped
        # body 的线速度存储在 cvel 中（6D: 3 旋转 + 3 平移）
        # 使用 subtree_linvel 获取线速度
        r_vel = env.data.body("foot").subtree_linvel
        l_vel = env.data.body("foot_left").subtree_linvel
        # 取 x(0) 和 z(2) 分量的范数
        r_speed = float(np.sqrt(r_vel[0]**2 + r_vel[2]**2))
        l_speed = float(np.sqrt(l_vel[0]**2 + l_vel[2]**2))
        return r_speed, l_speed

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

    def _contact_alternation_reward(self) -> float:
        """
        接触交替奖励。

        参考 LearningHumanoidWalking 中 calc_foot_frc_clock_reward
        的思路: 正常行走时，每个时刻应该有一只脚在地面（支撑相）、
        另一只脚在空中（摆动相），两只脚的接触力应呈反相关系。

        具体实现:
            - 着地的脚应有较大接触力，同时速度较低（支撑）
            - 抬起的脚应无接触力，同时速度较高（摆动）
            - 用力与速度的交叉乘积来度量交替程度

        返回:
            [0, 1] 区间内的奖励值。
        """
        r_frc, l_frc = self._get_foot_contact_forces()
        r_vel, l_vel = self._get_foot_velocities()

        # 归一化接触力: 机器人质量约 3.5kg，
        # 单脚最大承力约 3.5*9.8 ≈ 34N
        max_frc = 40.0
        r_frc_norm = min(r_frc / max_frc, 1.0)
        l_frc_norm = min(l_frc / max_frc, 1.0)

        # 归一化脚部速度
        max_vel = 2.0
        r_vel_norm = min(r_vel / max_vel, 1.0)
        l_vel_norm = min(l_vel / max_vel, 1.0)

        # 交叉得分: 右脚着地（高力低速）+ 左脚摆动（低力高速），
        # 或者反过来。两种情况取较大值。
        score_r_stance = r_frc_norm * l_vel_norm
        score_l_stance = l_frc_norm * r_vel_norm
        return max(score_r_stance, score_l_stance)

    def _foot_usage_reward(self) -> float:
        """
        双脚使用率奖励。

        追踪滑动窗口内两只脚的接触次数，奖励两脚均匀
        交替使用。如果一只脚长期不接触地面（被拖着走），
        使用率比值会很低，奖励接近零。

        这是解决"单腿拖行"问题的核心奖励: 即使瞬时交替
        得分不错，但长期只用一只脚仍会被惩罚。

        返回:
            [0, 1] 区间内的奖励值。
        """
        # 窗口不够长时不惩罚，给予充分的探索时间
        window = _WALK.foot_usage_window
        if len(self._right_contact_history) < window // 2:
            return 1.0

        r_count = sum(self._right_contact_history)
        l_count = sum(self._left_contact_history)
        total = r_count + l_count

        if total == 0:
            # 两只脚都没接触地面（腾空），不惩罚
            return 1.0

        # 使用率比值: 两脚各自占总接触的比例，
        # 取较小值。完美交替时 ratio=0.5，单脚时 ratio=0
        ratio = min(r_count, l_count) / total

        return tolerance(
            ratio,
            bounds=(_WALK.foot_usage_min_ratio, 1.0),
            margin=_WALK.foot_usage_min_ratio,
            value_at_margin=0,
            sigmoid="linear",
        )

    def _action_smoothness_reward(self) -> float:
        """
        动作平滑奖励。

        参考 LearningHumanoidWalking 中 calc_action_reward:
        惩罚相邻时间步之间动作的突变，使步态更自然连贯，
        减少关节抖动和不协调的急停急转。

        返回:
            [0, 1] 区间内的奖励值。
        """
        if self._prev_actions is None:
            return 1.0
        ctrl = self.single_agent_env.unwrapped.data.ctrl
        diff = np.abs(ctrl - self._prev_actions)
        penalty = 5.0 * np.mean(diff)
        return float(np.exp(-penalty))

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

        综合六个子奖励:
            - speed: 沿 x 正方向达到目标速度 1.0 m/s
            - height: 躯干高度保持在目标范围内
            - upright: 躯干保持直立姿态
            - alternation: 接触交替奖励，鼓励一脚支撑
                一脚摆动的正常步态
            - foot_usage: 双脚使用率，惩罚长期只用
                一只脚的拖行行为
            - smooth: 动作平滑，减少关节抖动

        乘法组合确保智能体必须同时满足所有约束:
        仅跑快但姿态不稳不会获得高分，
        仅站稳但不前进也不会获得高分，
        只用一只脚拖行也不会获得高分。

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
        alternation = self._contact_alternation_reward()
        foot_usage = self._foot_usage_reward()
        smooth = self._action_smoothness_reward()

        # 平滑奖励压缩至 [0.8, 1.0]，作为辅助约束
        smooth = (4 + smooth) / 5

        return (
            speed * height * upright
            * alternation * foot_usage * smooth
        )

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
