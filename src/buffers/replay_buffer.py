"""
多智能体多任务离线重放缓冲区。

参考 m3w-marl 的 ``WorldModelBuffer``，针对 ``MultiTaskMaMuJoCo``
+ ``ShareVecEnv`` 的接口做简化与现代化：

* **连续动作专用**：MaMuJoCo 全部为 ``Box`` 动作空间，因此不再保留
  ``available_actions`` / ``Discrete`` 分支，所有动作统一存为
  ``float32``，维度对齐 ``MultiTaskMaMuJoCo.max_action_dim``。
* **CTDE 友好**：同时存储每个智能体的局部观测 ``obs`` 与对应的共享
  状态 ``shared_obs``，方便 actor 用前者、critic 用后者。
* **跨环境步进**：沿用 m3w 的存放方式——同一时刻的 ``num_envs`` 条
  transition 连续插入，于是同一环境的相邻时间步刚好相隔
  ``num_envs`` 个槽位，:meth:`_next_indices` 利用这一步长在同一条
  episode 内向后游走，便于 n-step 回报与 horizon 序列采样。
* **两种采样接口**:

  - :meth:`sample` —— 随机采 ``batch_size`` 条，并按 ``n_step`` 折扣
    向前累加奖励，用于 SAC / TD3 风格的 critic 更新；
  - :meth:`sample_horizon` —— 随机采 ``batch_size`` 条起点，沿同一
    环境向后取连续 ``horizon`` 步，用于世界模型 / 多步 dynamics
    与 reward 训练。

约定的张量形状（``B`` = ``batch_size``，``H`` = ``horizon``，
``N_a`` = ``n_agents``）::

    obs / next_obs            : (B, N_a, obs_dim)
    shared_obs / next_shared  : (B, N_a, shared_state_dim)
    actions                   : (B, N_a, action_dim)
    rewards                   : (B, 1)            # 团队共享标量
    dones / terminations      : (B, 1)            # 环境级布尔
    sp_horizon_*              : (H, B, ...) 其余维度同上
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    """
    多智能体多任务离线重放缓冲区。

    存储 ``(obs, shared_obs, actions, reward, done, termination,
    next_obs, next_shared_obs)`` 元组，并提供单步 + n-step 折扣
    的随机采样以及定长 horizon 的轨迹片段采样。
    """

    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        n_agents: int,
        obs_dim: int,
        shared_state_dim: int,
        action_dim: int,
        n_step: int = 1,
        gamma: float = 0.99,
    ) -> None:
        """
        参数:
            buffer_size: 缓冲区可保存的 transition 总数。
            num_envs: 同时跑的并行环境数，决定 transition 的存放步长。
            n_agents: 智能体数量，对应 ``MultiTaskMaMuJoCo.max_n_agents``。
            obs_dim: 单个智能体的对齐观测维度（含任务 one-hot 前缀）。
            shared_state_dim: 单个智能体的共享中心化状态维度。
            action_dim: 单个智能体的动作维度，对应 ``max_action_dim``。
            n_step: ``sample`` 使用的 n-step 回报步数，``1`` 表示单步 TD。
            gamma: 折扣因子。

        异常:
            ValueError: 当 ``buffer_size`` 不是 ``num_envs`` 的整数倍时抛出。
        """
        if buffer_size % num_envs != 0:
            raise ValueError(
                f"buffer_size ({buffer_size}) 必须是 num_envs "
                f"({num_envs}) 的整数倍，以保证同一环境步长一致。"
            )
        if n_step < 1:
            raise ValueError(f"n_step 必须 >= 1，实际得到 {n_step}。")

        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.shared_state_dim = shared_state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma

        self.current_size: int = 0
        self.write_index: int = 0

        # 各张量的预分配
        self.obs = np.zeros(
            (buffer_size, n_agents, obs_dim),
            dtype=np.float32,
        )
        self.next_obs = np.zeros_like(self.obs)
        self.shared_obs = np.zeros(
            (buffer_size, n_agents, shared_state_dim),
            dtype=np.float32,
        )
        self.next_shared_obs = np.zeros_like(self.shared_obs)
        self.actions = np.zeros(
            (buffer_size, n_agents, action_dim),
            dtype=np.float32,
        )
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.bool_)
        self.terminations = np.zeros((buffer_size, 1), dtype=np.bool_)

        # 给 n-step / horizon 采样使用的"episode 终止标志"，
        # 每次 sample 前由 _refresh_end_flag 重新计算
        self._end_flag: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def insert(
        self,
        obs: np.ndarray,
        shared_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        terminations: np.ndarray,
        next_obs: np.ndarray,
        next_shared_obs: np.ndarray,
    ) -> None:
        """
        插入一组并行环境同一时刻的 transition。

        参数:
            obs: 形状 ``(num_envs, n_agents, obs_dim)``。
            shared_obs: 形状 ``(num_envs, n_agents, shared_state_dim)``。
            actions: 形状 ``(num_envs, n_agents, action_dim)``。
            rewards: 形状 ``(num_envs,)`` 或 ``(num_envs, 1)`` 的团队奖励。
            dones: 形状 ``(num_envs,)`` 或 ``(num_envs, 1)`` 的回合结束标志。
            terminations: 与 ``dones`` 同形状的真终止标志（区别于
                因步数限制触发的截断）。
            next_obs: 与 ``obs`` 同形状的下一时刻观测。
            next_shared_obs: 与 ``shared_obs`` 同形状的下一时刻共享状态。

        异常:
            ValueError: 当各张量第 0 维与 ``num_envs`` 不一致时抛出。
        """
        if obs.shape[0] != self.num_envs:
            raise ValueError(
                f"插入数据的第 0 维 {obs.shape[0]} 与 num_envs "
                f"{self.num_envs} 不一致。"
            )

        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs, 1)
        dones = np.asarray(dones, dtype=np.bool_).reshape(self.num_envs, 1)
        terminations = np.asarray(
            terminations,
            dtype=np.bool_,
        ).reshape(self.num_envs, 1)

        write_slots = (
            self.write_index + np.arange(self.num_envs)
        ) % self.buffer_size

        self.obs[write_slots] = obs
        self.shared_obs[write_slots] = shared_obs
        self.actions[write_slots] = actions
        self.rewards[write_slots] = rewards
        self.dones[write_slots] = dones
        self.terminations[write_slots] = terminations
        self.next_obs[write_slots] = next_obs
        self.next_shared_obs[write_slots] = next_shared_obs

        self.write_index = (self.write_index + self.num_envs) % self.buffer_size
        self.current_size = min(
            self.current_size + self.num_envs,
            self.buffer_size,
        )

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        """
        随机采样 ``batch_size`` 条 transition 并计算 n-step 折扣回报。

        参数:
            batch_size: 期望采样的 transition 条数。

        返回:
            字典，键含义如下:

            * ``obs`` / ``shared_obs``: ``(B, N_a, *)`` 起点状态；
            * ``actions``: ``(B, N_a, action_dim)`` 起点动作；
            * ``rewards``: ``(B, 1)`` 累加后的 n-step 折扣回报；
            * ``gammas``: ``(B, 1)`` 实际折扣系数 ``γ^n``，遇到回合结束
              会被截断到对应步数；
            * ``dones`` / ``terminations``: ``(B, 1)`` 末步标志；
            * ``next_obs`` / ``next_shared_obs``: ``(B, N_a, *)`` 末步
              对应的下一状态；
            * ``next_obs_one_step`` / ``next_shared_obs_one_step``:
              起点的单步下一状态，便于世界模型 + n-step critic 联合训练。

        异常:
            ValueError: 当 ``batch_size`` 大于 ``current_size`` 时抛出。
        """
        if batch_size > self.current_size:
            raise ValueError(
                f"batch_size ({batch_size}) 不能大于当前缓冲区大小 "
                f"({self.current_size})。"
            )

        self._refresh_end_flag()
        start_indices = np.random.choice(
            self.current_size,
            size=batch_size,
            replace=False,
        )

        # 沿同一环境向后游走 n_step - 1 次
        rolling_indices = [start_indices]
        for _ in range(self.n_step - 1):
            rolling_indices.append(self._next_indices(rolling_indices[-1]))

        last_indices = rolling_indices[-1]

        # 累加 n-step 折扣回报，遇到回合结束则截断
        rewards_n_step = np.zeros((batch_size, 1), dtype=np.float32)
        effective_n = np.full(batch_size, self.n_step, dtype=np.int64)
        for step_index in range(self.n_step - 1, -1, -1):
            indices_at_step = rolling_indices[step_index]
            ended = self._end_flag[indices_at_step]
            effective_n[ended] = step_index + 1
            rewards_n_step[ended] = 0.0
            rewards_n_step = (
                self.rewards[indices_at_step] + self.gamma * rewards_n_step
            )

        gamma_powers = np.power(self.gamma, np.arange(self.n_step + 1))
        gammas = gamma_powers[effective_n].reshape(batch_size, 1)

        return {
            "obs": self.obs[start_indices],
            "shared_obs": self.shared_obs[start_indices],
            "actions": self.actions[start_indices],
            "rewards": rewards_n_step,
            "gammas": gammas,
            "dones": self.dones[last_indices],
            "terminations": self.terminations[last_indices],
            "next_obs": self.next_obs[last_indices],
            "next_shared_obs": self.next_shared_obs[last_indices],
            "next_obs_one_step": self.next_obs[start_indices],
            "next_shared_obs_one_step": self.next_shared_obs[start_indices],
            "rewards_one_step": self.rewards[start_indices],
        }

    def sample_horizon(
        self,
        batch_size: int,
        horizon: int,
    ) -> dict[str, np.ndarray]:
        """
        随机采 ``batch_size`` 条起点，沿同一环境取连续 ``horizon`` 步。

        参数:
            batch_size: 起点条数。
            horizon: 序列长度。

        返回:
            字典，每项形状均带 ``(horizon, batch_size, ...)`` 的最外层，
            包含 ``obs / shared_obs / actions / rewards / dones /
            terminations / next_obs / next_shared_obs``。

        异常:
            ValueError: 当 ``batch_size`` 大于 ``current_size`` 或
                ``horizon < 1`` 时抛出。
        """
        if batch_size > self.current_size:
            raise ValueError(
                f"batch_size ({batch_size}) 不能大于当前缓冲区大小 "
                f"({self.current_size})。"
            )
        if horizon < 1:
            raise ValueError(f"horizon 必须 >= 1，实际得到 {horizon}。")

        self._refresh_end_flag()
        start_indices = np.random.choice(
            self.current_size,
            size=batch_size,
            replace=False,
        )

        rolling_indices = [start_indices]
        for _ in range(horizon - 1):
            rolling_indices.append(self._next_indices(rolling_indices[-1]))
        stacked = np.stack(rolling_indices, axis=0)  # (horizon, batch_size)

        return {
            "obs": self.obs[stacked],
            "shared_obs": self.shared_obs[stacked],
            "actions": self.actions[stacked],
            "rewards": self.rewards[stacked],
            "dones": self.dones[stacked],
            "terminations": self.terminations[stacked],
            "next_obs": self.next_obs[stacked],
            "next_shared_obs": self.next_shared_obs[stacked],
        }

    # ------------------------------------------------------------------
    # 杂项
    # ------------------------------------------------------------------

    def get_mean_reward(self) -> float:
        """返回缓冲区中所有已写入 reward 的均值。"""
        if self.current_size == 0:
            return 0.0
        return float(self.rewards[: self.current_size].mean())

    def __len__(self) -> int:
        """返回当前已写入的 transition 数。"""
        return self.current_size

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _refresh_end_flag(self) -> None:
        """
        重新计算 ``_end_flag``。

        除了真正的 episode 结束位置外，每个并行环境最近一次写入的
        位置（尚未拥有"下一步"）也被视为终止，避免 ``_next_indices``
        游走到空白槽位上。
        """
        end_flag = self.dones[: self.current_size, 0].copy()
        unfinished = (
            self.write_index - 1 - np.arange(self.num_envs) + self.current_size
        ) % self.current_size
        end_flag[unfinished] = True
        self._end_flag = end_flag

    def _next_indices(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        给定一组 transition 下标，返回**同一环境**的下一时间步下标。

        如果当前位置已经是 episode 终止或缓冲区写头之前的最新一步，
        则原地停留，避免越界或跨 episode。
        """
        assert self._end_flag is not None
        stride = self.num_envs * (1 - self._end_flag[indices].astype(np.int64))
        return (indices + stride) % self.current_size
