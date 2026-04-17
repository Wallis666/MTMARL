"""
HalfCheetah 多任务环境适配器模块。

将 HalfCheetahMultiTask（PettingZoo Gymnasium 接口）适配为
baselines 训练框架所需的接口格式。
"""

from typing import Any

import numpy as np
from gymnasium import spaces

from src.envs.mamujoco.tasks.cheetah import (
    HalfCheetahMultiTask,
)


class HalfCheetahMultiTaskHARL:
    """
    HalfCheetah 多任务环境的 HARL 适配器。

    将 PettingZoo 的 dict 接口转换为 baselines 框架期望的
    数组接口，桥接以下差异:
        - step: 5 元组 dict → 6 元组 array
        - reset: 2 元组 dict → 3 元组 array
        - done: terminated + truncated → 单个 done
        - 智能体索引: 字符串键 → 整数索引
        - seed: reset(seed=...) → env.seed(s)
        - 属性: num_agents → n_agents
        - 新增: share_observation_space、available_actions
    """

    def __init__(
        self,
        env_args: dict[str, Any],
    ) -> None:
        """
        初始化适配器。

        参数:
            env_args: 环境参数字典，需包含 agent_conf，
                可选 agent_obsk、task、episode_limit、
                render_mode 等。
        """
        self._env = HalfCheetahMultiTask(
            agent_conf=env_args.get("agent_conf"),
            agent_obsk=env_args.get("agent_obsk", 1),
            render_mode=env_args.get("render_mode"),
        )

        # 设置初始任务
        task = env_args.get("task", "run_fwd")
        self._env.set_task(task)

        # 单回合最大步数
        self.episode_limit = int(
            env_args.get("episode_limit", 1000),
        )

        # 缓存智能体名称列表（有序）
        self._agents = sorted(
            self._env.possible_agents,
        )
        self.n_agents = len(self._agents)

        # 构建空间列表，按智能体索引访问
        self.observation_space = [
            self._env.observation_space(agent)
            for agent in self._agents
        ]
        self.action_space = [
            self._env.action_space(agent)
            for agent in self._agents
        ]

        # 共享观测空间: 使用底层 MuJoCo 完整状态
        state_dim = int(
            np.asarray(self._env.state()).shape[0],
        )
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]

        self._seed_value = None
        self._steps = 0

    # ----------------------------------------------------------
    # 核心接口
    # ----------------------------------------------------------

    def seed(
        self,
        seed: int,
    ) -> None:
        """
        记录种子，将在下一次 reset 时生效。

        参数:
            seed: 随机种子值。
        """
        self._seed_value = int(seed)

    def reset(self) -> tuple:
        """
        重置环境。

        返回:
            (obs, share_obs, available_actions) 三元组:
                obs: (n_agents, obs_dim) 各智能体观测。
                share_obs: (n_agents, state_dim) 共享
                    观测，所有智能体共享同一全局状态。
                available_actions: 可用动作掩码或 None。
        """
        seed = self._seed_value
        self._seed_value = None

        if seed is None:
            obs_dict, _ = self._env.reset()
        else:
            obs_dict, _ = self._env.reset(seed=seed)

        self._steps = 0
        obs = self._dict_to_array(obs_dict)
        share_obs = self._build_share_obs()
        available_actions = self._get_available_actions()

        return obs, share_obs, available_actions

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple:
        """
        执行一步交互。

        参数:
            actions: (n_agents, action_dim) 各智能体动作。

        返回:
            (obs, share_obs, rewards, dones, infos,
             available_actions) 六元组。
        """
        # 数组动作 → 字典动作
        action_dict = {
            agent: np.asarray(
                actions[i], dtype=np.float32,
            )
            for i, agent in enumerate(self._agents)
        }

        (
            obs_dict,
            reward_dict,
            term_dict,
            trunc_dict,
            info_dict,
        ) = self._env.step(action_dict)
        self._steps += 1

        obs = self._dict_to_array(obs_dict)
        share_obs = self._build_share_obs()

        # 各智能体奖励 → (n_agents, 1)
        rewards = np.array(
            [
                [float(reward_dict[agent])]
                for agent in self._agents
            ],
            dtype=np.float32,
        )

        # term = 真实终止，trunc = 时间截断
        # bad_transition=True 表示被截断（不要用 0 自举）
        term_any = any(
            bool(term_dict[a]) for a in self._agents
        )
        trunc_any = any(
            bool(trunc_dict[a]) for a in self._agents
        )
        time_limit = self._steps >= self.episode_limit
        done_flag = (
            term_any or trunc_any or time_limit
        )
        bad_transition = (
            (trunc_any or time_limit) and not term_any
        )

        dones = np.array(
            [done_flag] * self.n_agents, dtype=bool,
        )

        # infos 转为列表格式，附加 bad_transition
        infos = []
        for agent in self._agents:
            agent_info = dict(
                info_dict.get(agent, {}),
            )
            agent_info["bad_transition"] = (
                bad_transition
            )
            agent_info["task"] = self._env.task
            infos.append(agent_info)

        available_actions = self._get_available_actions()

        return (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
        )

    def close(self) -> None:
        """关闭环境。"""
        self._env.close()

    def render(
        self,
        mode: str = "human",
    ) -> Any:
        """
        渲染环境。

        参数:
            mode: 渲染模式（gymnasium 以初始化时的
                render_mode 为准，此参数仅为兼容接口）。

        返回:
            渲染结果（取决于模式）。
        """
        del mode
        return self._env.render()

    # ----------------------------------------------------------
    # 任务切换（透传）
    # ----------------------------------------------------------

    def set_task(
        self,
        task,
    ) -> None:
        """
        切换当前任务。

        参数:
            task: 任务名称或任务索引。
        """
        self._env.set_task(task)

    @property
    def task(self) -> str:
        """返回当前任务名称。"""
        return self._env.task

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _dict_to_array(
        self,
        obs_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        将智能体字典观测转换为数组。

        参数:
            obs_dict: 键为智能体名称的观测字典。

        返回:
            (n_agents, obs_dim) 的观测数组。
        """
        return np.array(
            [obs_dict[agent] for agent in self._agents],
            dtype=np.float32,
        )

    def _build_share_obs(self) -> np.ndarray:
        """
        构建共享观测，使用底层 MuJoCo 完整状态。

        返回:
            (n_agents, state_dim) 共享观测，
            每个智能体看到相同的全局状态。
        """
        state = np.asarray(
            self._env.state(), dtype=np.float32,
        )
        return np.tile(state, (self.n_agents, 1))

    def _get_available_actions(self):
        """
        获取可用动作掩码。

        返回:
            连续动作空间返回 None，离散动作空间返回
            全 1 掩码。
        """
        space = self.action_space[0]
        if space.__class__.__name__ == "Discrete":
            return np.ones(
                (self.n_agents, space.n),
                dtype=np.float32,
            )
        return None
