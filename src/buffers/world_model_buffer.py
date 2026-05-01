"""World Model 经验回放缓冲区模块。

提供离线策略经验回放缓冲区，支持单步随机采样和
连续多步轨迹段采样，分别用于 Critic 更新和
World Model 训练。
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from src.utils.env import (
    get_shape_from_act_space,
    get_shape_from_obs_space,
)


class WorldModelBuffer:
    """World Model 离线策略经验回放缓冲区。

    使用环形缓冲区存储多智能体交互数据，支持两种
    采样方式:

    - ``sample``: 随机采样单步转移，用于 Critic
      的 n-step TD 更新。
    - ``sample_horizon``: 采样连续轨迹段，用于
      World Model（编码器/动力学/奖励模型）训练。
    """

    def __init__(
        self,
        args: dict,
        share_obs_space: spaces.Space,
        num_agents: int,
        obs_spaces: list[spaces.Space],
        act_spaces: list[spaces.Space],
    ) -> None:
        """
        初始化经验回放缓冲区。

        参数:
            args: 参数字典，需包含:
                - buffer_size: 缓冲区最大容量。
                - batch_size: 采样批量大小。
                - n_step: n 步回报的步数。
                - n_rollout_threads: 并行环境数量。
                - gamma: 折扣因子。
            share_obs_space: 共享观测空间。
            num_agents: 智能体数量。
            obs_spaces: 各智能体的观测空间列表。
            act_spaces: 各智能体的动作空间列表。
        """
        self.buffer_size = args["buffer_size"]
        self.batch_size = args["batch_size"]
        self.n_step = args["n_step"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.gamma = args["gamma"]
        self.cur_size = 0
        self.idx = 0
        self.num_agents = num_agents
        self.act_spaces = act_spaces

        # 解析空间形状
        self.share_obs_shape = get_shape_from_obs_space(
            share_obs_space,
        )
        if isinstance(self.share_obs_shape[-1], list):
            self.share_obs_shape = (
                self.share_obs_shape[:1]
            )

        obs_shapes: list[tuple] = []
        act_shapes: list[int] = []
        for agent_id in range(num_agents):
            obs_shape = get_shape_from_obs_space(
                obs_spaces[agent_id],
            )
            if isinstance(obs_shape[-1], list):
                obs_shape = obs_shape[:1]
            obs_shapes.append(obs_shape)
            act_shapes.append(
                get_shape_from_act_space(
                    act_spaces[agent_id],
                ),
            )

        # 各智能体观测缓冲区
        self.obs: list[np.ndarray] = []
        self.next_obs: list[np.ndarray] = []
        for agent_id in range(num_agents):
            self.obs.append(np.zeros(
                (self.buffer_size,
                 *obs_shapes[agent_id]),
                dtype=np.float32,
            ))
            self.next_obs.append(np.zeros(
                (self.buffer_size,
                 *obs_shapes[agent_id]),
                dtype=np.float32,
            ))

        # 各智能体有效转移标记
        self.valid_transitions: list[np.ndarray] = []
        for agent_id in range(num_agents):
            self.valid_transitions.append(np.ones(
                (self.buffer_size, 1),
                dtype=np.float32,
            ))

        # 各智能体动作及可用动作缓冲区
        self.actions: list[np.ndarray] = []
        self.available_actions: list[np.ndarray] = []
        self.next_available_actions: (
            list[np.ndarray]
        ) = []
        for agent_id in range(num_agents):
            self.actions.append(np.zeros(
                (self.buffer_size,
                 act_shapes[agent_id]),
                dtype=np.float32,
            ))
            act_cls = (
                act_spaces[agent_id]
                .__class__.__name__
            )
            if act_cls == "Discrete":
                n_acts = act_spaces[agent_id].n
                self.available_actions.append(
                    np.zeros(
                        (self.buffer_size, n_acts),
                        dtype=np.float32,
                    ),
                )
                self.next_available_actions.append(
                    np.zeros(
                        (self.buffer_size, n_acts),
                        dtype=np.float32,
                    ),
                )

        # 共享观测缓冲区
        self.share_obs = np.zeros(
            (self.buffer_size,
             *self.share_obs_shape),
            dtype=np.float32,
        )
        self.next_share_obs = np.zeros(
            (self.buffer_size,
             *self.share_obs_shape),
            dtype=np.float32,
        )

        # 奖励与终止标志
        self.rewards = np.zeros(
            (self.buffer_size, 1), dtype=np.float32,
        )
        self.dones = np.full(
            (self.buffer_size, 1), False,
        )
        self.terms = np.full(
            (self.buffer_size, 1), False,
        )

    # -------------------------------------------------------
    # 数据写入
    # -------------------------------------------------------

    def insert(
        self,
        data: tuple,
    ) -> None:
        """
        向缓冲区中插入一批转移数据。

        当写入位置超出缓冲区末尾时自动环形覆盖。

        参数:
            data: 包含以下元素的元组:
                (share_obs, obs, actions,
                 available_actions, reward, done,
                 valid_transitions, term,
                 next_share_obs, next_obs,
                 next_available_actions)。
        """
        (
            share_obs,
            obs,
            actions,
            available_actions,
            reward,
            done,
            valid_transitions,
            term,
            next_share_obs,
            next_obs,
            next_available_actions,
        ) = data
        length = share_obs.shape[0]

        if self.idx + length <= self.buffer_size:
            self._write_slice(
                self.idx,
                self.idx + length,
                share_obs, obs, actions,
                available_actions, reward, done,
                valid_transitions, term,
                next_share_obs, next_obs,
                next_available_actions,
            )
        else:
            # 分两段写入
            len1 = self.buffer_size - self.idx
            len2 = length - len1
            self._write_slice(
                self.idx,
                self.buffer_size,
                share_obs[:len1], [o[:len1] for o in obs],
                [a[:len1] for a in actions],
                ([a[:len1] for a in available_actions]
                 if available_actions else None),
                reward[:len1], done[:len1],
                [v[:len1] for v in valid_transitions],
                term[:len1],
                next_share_obs[:len1],
                [o[:len1] for o in next_obs],
                ([a[:len1] for a in next_available_actions]
                 if next_available_actions else None),
            )
            self._write_slice(
                0,
                len2,
                share_obs[len1:], [o[len1:] for o in obs],
                [a[len1:] for a in actions],
                ([a[len1:] for a in available_actions]
                 if available_actions else None),
                reward[len1:], done[len1:],
                [v[len1:] for v in valid_transitions],
                term[len1:],
                next_share_obs[len1:],
                [o[len1:] for o in next_obs],
                ([a[len1:] for a in next_available_actions]
                 if next_available_actions else None),
            )

        self.idx = (
            (self.idx + length) % self.buffer_size
        )
        self.cur_size = min(
            self.cur_size + length, self.buffer_size,
        )

    def _write_slice(
        self,
        start: int,
        end: int,
        share_obs: np.ndarray,
        obs: list[np.ndarray],
        actions: list[np.ndarray],
        available_actions: list[np.ndarray] | None,
        reward: np.ndarray,
        done: np.ndarray,
        valid_transitions: list[np.ndarray],
        term: np.ndarray,
        next_share_obs: np.ndarray,
        next_obs: list[np.ndarray],
        next_available_actions: (
            list[np.ndarray] | None
        ),
    ) -> None:
        """
        将数据写入缓冲区的指定切片区间。

        参数:
            start: 写入起始索引。
            end: 写入结束索引（不含）。
            share_obs: 共享观测。
            obs: 各智能体观测列表。
            actions: 各智能体动作列表。
            available_actions: 可用动作掩码列表。
            reward: 奖励。
            done: 终止标志。
            valid_transitions: 有效转移标记列表。
            term: 截断标志。
            next_share_obs: 下一步共享观测。
            next_obs: 各智能体下一步观测列表。
            next_available_actions: 下一步可用动作列表。
        """
        s, e = start, end
        self.share_obs[s:e] = share_obs.copy()
        self.rewards[s:e] = reward.copy()
        self.dones[s:e] = done.copy()
        self.terms[s:e] = term.copy()
        self.next_share_obs[s:e] = (
            next_share_obs.copy()
        )
        for agent_id in range(self.num_agents):
            self.obs[agent_id][s:e] = (
                obs[agent_id].copy()
            )
            self.actions[agent_id][s:e] = (
                actions[agent_id].copy()
            )
            self.valid_transitions[agent_id][s:e] = (
                valid_transitions[agent_id].copy()
            )
            self.next_obs[agent_id][s:e] = (
                next_obs[agent_id].copy()
            )
            act_cls = (
                self.act_spaces[agent_id]
                .__class__.__name__
            )
            if act_cls == "Discrete":
                self.available_actions[
                    agent_id
                ][s:e] = (
                    available_actions[agent_id].copy()
                )
                self.next_available_actions[
                    agent_id
                ][s:e] = (
                    next_available_actions[
                        agent_id
                    ].copy()
                )

    # -------------------------------------------------------
    # 单步采样（Critic 更新）
    # -------------------------------------------------------

    def sample(
        self,
        indices: np.ndarray | None = None,
    ) -> tuple:
        """
        随机采样单步转移并计算 n-step 累积回报。

        用于 Critic 的 off-policy 更新。当缓冲区中存储
        了连续多步数据时，自动沿时间维度展开计算 n-step
        折扣回报和对应的 gamma 值。

        参数:
            indices: 可选的采样索引数组。为 None 时
                随机采样 batch_size 个样本。

        返回:
            包含以下元素的元组:
            - share_obs: (batch, *dim)
            - obs: (n_agents, batch, *dim)
            - actions: (n_agents, batch, *dim)
            - available_actions: (n_agents, batch,
              *dim) 或 None
            - reward: (batch, 1) n-step 累积奖励
            - done: (batch, 1)
            - valid_transitions: (n_agents, batch, 1)
            - term: (batch, 1)
            - next_share_obs: (batch, *dim)
            - next_obs: (n_agents, batch, *dim)
            - next_available_actions:
              (n_agents, batch, *dim) 或 None
            - gamma: (batch, 1) n-step 折扣因子
            - next_obs_1step: (n_agents, batch, *dim)
            - reward_1step: (batch, 1)
        """
        self._update_end_flag()

        if indices is None:
            indices = torch.randperm(
                self.cur_size,
            ).numpy()[:self.batch_size]

        # 起始步数据
        sp_share_obs = self.share_obs[indices]
        sp_obs = np.array([
            self.obs[i][indices]
            for i in range(self.num_agents)
        ])
        sp_actions = np.array([
            self.actions[i][indices]
            for i in range(self.num_agents)
        ])
        sp_valid_transitions = np.array([
            self.valid_transitions[i][indices]
            for i in range(self.num_agents)
        ])

        is_discrete = (
            self.act_spaces[0].__class__.__name__
            == "Discrete"
        )
        if is_discrete:
            sp_available_actions = np.array([
                self.available_actions[i][indices]
                for i in range(self.num_agents)
            ])
        else:
            sp_available_actions = None

        # 沿 n 步展开索引
        idx_chain = [indices]
        for _ in range(self.n_step - 1):
            idx_chain.append(
                self._next_indices(idx_chain[-1]),
            )

        # 末端步数据
        last = idx_chain[-1]
        sp_done = self.dones[last]
        sp_term = self.terms[last]
        sp_next_share_obs = self.next_share_obs[last]
        sp_next_obs = np.array([
            self.next_obs[i][last]
            for i in range(self.num_agents)
        ])
        if is_discrete:
            sp_next_available_actions = np.array([
                self.next_available_actions[i][last]
                for i in range(self.num_agents)
            ])
        else:
            sp_next_available_actions = None

        # n-step 累积奖励
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = (
                gamma_buffer[i - 1] * self.gamma
            )
        sp_reward = np.zeros(
            (self.batch_size, 1),
        )
        gammas = np.full(
            self.batch_size, self.n_step,
        )
        for n in range(self.n_step - 1, -1, -1):
            now = idx_chain[n]
            ended = self._end_flag[now] > 0
            gammas[ended] = n + 1
            sp_reward[ended] = 0.0
            sp_reward = (
                self.rewards[now]
                + self.gamma * sp_reward
            )
        sp_gamma = gamma_buffer[gammas].reshape(
            self.batch_size, 1,
        )

        # 1-step 数据（用于 world model 训练）
        sp_next_obs_1step = np.array([
            self.next_obs[i][idx_chain[0]]
            for i in range(self.num_agents)
        ])
        sp_reward_1step = self.rewards[idx_chain[0]]

        return (
            sp_share_obs,
            sp_obs,
            sp_actions,
            sp_available_actions,
            sp_reward,
            sp_done,
            sp_valid_transitions,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            sp_next_available_actions,
            sp_gamma,
            sp_next_obs_1step,
            sp_reward_1step,
        )

    # -------------------------------------------------------
    # 轨迹段采样（World Model 训练）
    # -------------------------------------------------------

    def sample_horizon(
        self,
        horizon: int | None = None,
        t0: np.ndarray | None = None,
    ) -> tuple:
        """
        采样连续多步轨迹段。

        用于 World Model（编码器、动力学模型、奖励模型）
        的序列训练。采样结果按时间维度排列，遇到回合
        终止时停留在终止步不再前进。

        参数:
            horizon: 轨迹段长度，默认使用 n_step。
            t0: 轨迹段起始索引数组，默认随机采样。

        返回:
            包含以下元素的元组:
            - share_obs: (horizon, batch, *dim)
            - obs: (n_agents, horizon, batch, *dim)
            - actions: (n_agents, horizon, batch, *dim)
            - available_actions: None（连续空间）
            - reward: (horizon, batch, 1)
            - done: (horizon, batch, 1)
            - valid_transitions:
              (n_agents, horizon, batch, 1)
            - term: (horizon, batch, 1)
            - next_share_obs: (horizon, batch, *dim)
            - next_obs:
              (n_agents, horizon, batch, *dim)
            - next_available_actions: None（连续空间）
            - gamma: (horizon, batch, 1)
        """
        if horizon is None:
            horizon = self.n_step
        self._update_end_flag()

        if t0 is None:
            t0 = torch.randperm(
                self.cur_size,
            ).numpy()[:self.batch_size]

        # 沿时间维度展开索引
        idx_chain = [t0]
        for _ in range(horizon - 1):
            idx_chain.append(
                self._next_indices(idx_chain[-1]),
            )

        # 按时间步收集数据
        sp_share_obs = np.array([
            self.share_obs[t] for t in idx_chain
        ])
        sp_obs = np.array([
            np.array([
                self.obs[i][t] for t in idx_chain
            ])
            for i in range(self.num_agents)
        ])
        sp_actions = np.array([
            np.array([
                self.actions[i][t] for t in idx_chain
            ])
            for i in range(self.num_agents)
        ])
        sp_valid_transitions = np.array([
            np.array([
                self.valid_transitions[i][t]
                for t in idx_chain
            ])
            for i in range(self.num_agents)
        ])
        sp_reward = np.array([
            self.rewards[t] for t in idx_chain
        ])
        sp_done = np.array([
            self.dones[t] for t in idx_chain
        ])
        sp_term = np.array([
            self.terms[t] for t in idx_chain
        ])
        sp_next_share_obs = np.array([
            self.next_share_obs[t] for t in idx_chain
        ])
        sp_next_obs = np.array([
            np.array([
                self.next_obs[i][t] for t in idx_chain
            ])
            for i in range(self.num_agents)
        ])
        sp_gamma = np.full(
            (horizon, self.batch_size, 1),
            self.gamma,
        )

        return (
            sp_share_obs,
            sp_obs,
            sp_actions,
            None,
            sp_reward,
            sp_done,
            sp_valid_transitions,
            sp_term,
            sp_next_share_obs,
            sp_next_obs,
            None,
            sp_gamma,
        )

    # -------------------------------------------------------
    # 统计
    # -------------------------------------------------------

    def get_mean_rewards(self) -> float:
        """返回缓冲区中已有数据的平均奖励。"""
        return float(
            np.mean(self.rewards[:self.cur_size]),
        )

    # -------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------

    def _next_indices(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        获取下一步索引。

        若当前步为回合结束或未完成的最新步，则索引
        不前进（停留在原位）。

        参数:
            indices: 当前步索引数组。

        返回:
            下一步索引数组。
        """
        return (
            indices
            + (1 - self._end_flag[indices])
            * self.n_rollout_threads
        ) % self.buffer_size

    def _update_end_flag(self) -> None:
        """
        更新回合结束标志。

        结束标志在以下位置为 True:
        - 回合真正终止的步骤。
        - 当前未完成回合的最新步骤（防止跨回合
          采样到未来数据）。
        """
        unfinished = (
            self.idx
            - np.arange(self.n_rollout_threads)
            - 1 + self.cur_size
        ) % self.cur_size
        self._end_flag = self.dones.copy().squeeze()
        self._end_flag[unfinished] = True
