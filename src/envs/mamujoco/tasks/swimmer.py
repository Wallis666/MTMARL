"""
Swimmer 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供多种游泳任务的自定义奖励函数，支持在任务间动态切换。
"""

from typing import Any

import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import (
    MultiAgentMujocoEnv,
)
from numpy.typing import NDArray


# ------------------------------------------------------------------
# 环境类
# ------------------------------------------------------------------


class SwimmerMultiTask(MultiAgentMujocoEnv):
    """
    Swimmer 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Swimmer，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    奖励基于归一化相位正弦值 sin(φ)：
        - swim_fwd: 奖励 sin(φ) → +1（φ → +π/2，
            joint1 领先，正向行波）
        - swim_bwd: 奖励 sin(φ) → -1（φ → -π/2，
            joint2 领先，反向行波）

    支持的任务集:
        - swim_fwd: 正向游泳
        - swim_bwd: 反向游泳
    """

    TASKS: list[str] = [
        "swim_fwd",
        "swim_bwd",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Swimmer 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "2x1" 表示
                2 个智能体各控制 1 个关节。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        # 默认 local_categories 对邻居只共享 qpos，
        # 这里覆盖为所有深度都共享 qpos+qvel，
        # 使每个 agent 能观测到对方关节的角速度。
        local_cat = [["qpos", "qvel"]] * (agent_obsk + 1)
        super().__init__(
            scenario="Swimmer",
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            local_categories=local_cat,
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
        task_reward = self._compute_reward()
        rewards = {agent: task_reward for agent in obs}
        # 仅在 human 渲染模式下打印，不影响训练
        if self._render_mode == "human":
            sin_phi = self._sin_phase()
            print(
                f"\rtask={self.task:<10} "
                f"sin(φ)={sin_phi:+.4f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        return obs, rewards, terms, truncs, infos

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _sin_phase(self) -> float:
        """
        估计 joint1 相对于 joint2 的相位差正弦值。

        利用归一化交叉积消除振幅和频率的影响：

            sin(φ) ≈ (q1·dq2 - q2·dq1)
                      / sqrt((q1² + q2²)(dq1² + dq2²))

        当两关节振幅相近（A1 ≈ A2）时，此估计在稳态下
        精确等于 sin(φ)：
            - φ = +π/2 → +1（joint1 领先，正向行波）
            - φ = -π/2 → -1（joint2 领先，反向行波）
            - φ = 0 或 π →  0（无行波）

        返回:
            [-1, 1] 区间内的标量。关节静止时返回 0。
        """
        data = self.single_agent_env.unwrapped.data
        q1 = data.qpos[3]   # motor1_rot 关节角
        q2 = data.qpos[4]   # motor2_rot 关节角
        dq1 = data.qvel[3]  # motor1_rot 角速度
        dq2 = data.qvel[4]  # motor2_rot 角速度

        cross = q1 * dq2 - q2 * dq1
        denom = np.sqrt(
            (q1 ** 2 + q2 ** 2)
            * (dq1 ** 2 + dq2 ** 2)
        )
        if denom < 1e-8:
            return 0.0
        return float(np.clip(cross / denom, -1.0, 1.0))

    # ------------------------------------------------------------------
    # 奖励分发
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        根据当前任务计算奖励。

        奖励 = max(±sin(φ), 0)，直接度量相位差接近
        ±π/2 的程度：
            - swim_fwd: 奖励 sin(φ) → +1（φ → +π/2）
            - swim_bwd: 奖励 sin(φ) → -1（φ → -π/2）

        返回:
            当前任务对应的标量奖励值。

        异常:
            NotImplementedError: 当前任务未实现时抛出。
        """
        sin_phi = self._sin_phase()
        task = self.task
        if task == "swim_fwd":
            return self._swim_fwd_reward(sin_phi)
        elif task == "swim_bwd":
            return self._swim_bwd_reward(sin_phi)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _swim_fwd_reward(
        self,
        sin_phi: float,
    ) -> float:
        """
        正向游泳奖励。

        奖励归一化相位正弦值的正部，鼓励 joint1 领先
        joint2 达到 π/2 相位差，形成正向行波。

        参数:
            sin_phi: 归一化相位正弦值。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return float(max(sin_phi, 0.0))

    def _swim_bwd_reward(
        self,
        sin_phi: float,
    ) -> float:
        """
        反向游泳奖励。

        奖励归一化相位正弦值的负部，鼓励 joint2 领先
        joint1 达到 π/2 相位差，形成反向行波。

        参数:
            sin_phi: 归一化相位正弦值。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return float(max(-sin_phi, 0.0))
