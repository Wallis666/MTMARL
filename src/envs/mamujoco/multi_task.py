"""
跨机器人多任务 MaMuJoCo 环境总线。

将若干异构的多智能体机器人环境（每个机器人内部又包含若干子任务）
统一打包成同一个对外环境对象，对外暴露形状对齐、可任意切换任务的
``reset`` / ``step`` / ``reset_task`` 接口。

主要解决以下问题：

* 不同机器人的观测维度、动作维度、智能体数量都不相同，需要统一对齐
  到所有域的最大值，便于上层 MARL 算法使用同一组张量形状。
* 在每个智能体的观测前拼接 **任务 one-hot**，让策略网络可以区分当前
  正在执行哪个任务。
* 提供按索引或按名字切换任务的入口，便于 Meta-RL / 多任务训练循环。

支持通过 YAML 配置文件描述任务集合，例如::

    domains:
      half_cheetah:
        tasks: [run, run_backwards, jump]
      humanoid:
        tasks: [stand, walk, run]

参考实现:
    refs/m3w-marl/m3w/envs/mujoco/multitask.py

参考文档:
    https://robotics.farama.org/envs/MaMuJoCo/
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv

from src.envs.mamujoco.tasks.cheetah import HalfCheetahMultiTask
from src.envs.mamujoco.tasks.humanoid import HumanoidMultiTask

# 域名称（YAML 配置中使用的 key）到具体环境类的注册表
DOMAIN_REGISTRY: dict[str, type[ParallelEnv]] = {
    "half_cheetah": HalfCheetahMultiTask,
    "humanoid": HumanoidMultiTask,
}


class MultiTaskMaMuJoCo:
    """
    跨机器人多任务 MaMuJoCo 总线环境。

    内部为每个域创建一个对应的 ``ParallelEnv`` 实例，并维护一个扁平化
    的任务列表。``reset`` / ``step`` 始终作用在 **当前激活任务** 所属
    的子环境上；上层算法看到的观测、动作、奖励都已对齐到所有域的
    最大尺寸。
    """

    def __init__(
        self,
        domains: dict[str, dict[str, Any]],
        episode_limit: int | None = None,
    ) -> None:
        """
        初始化总线环境。

        参数:
            domains: 域配置字典。键为 ``DOMAIN_REGISTRY`` 中的域名，
                值为该域的关键字参数（必须包含 ``tasks`` 列表）。例如::

                    {
                        "half_cheetah": {"tasks": ["run", "jump"]},
                        "humanoid": {"tasks": ["stand", "walk"]},
                    }

                除 ``tasks`` 外的其他键值会作为 ``**kwargs`` 透传给
                对应的环境构造函数。
            episode_limit: 单回合最大步数。``None`` 表示不在总线层做限制。

        异常:
            ValueError: 当 ``domains`` 为空、或包含未注册的域名、或
                某个域缺少 ``tasks`` 字段时抛出。
        """
        if not domains:
            raise ValueError("`domains` 至少需要包含一个域。")

        self.domain_configs: dict[str, dict[str, Any]] = dict(domains)
        self.episode_limit = episode_limit

        # ----- 1. 为每个域创建底层环境实例 -----
        self.domain_names: list[str] = list(self.domain_configs.keys())
        self.domain_envs: list[ParallelEnv] = []
        for domain_name in self.domain_names:
            if domain_name not in DOMAIN_REGISTRY:
                raise ValueError(
                    f"未知的域 {domain_name!r}，可选: "
                    f"{list(DOMAIN_REGISTRY)}。"
                )
            cfg = dict(self.domain_configs[domain_name])
            if "tasks" not in cfg:
                raise ValueError(
                    f"域 {domain_name!r} 的配置中缺少 `tasks` 字段。"
                )
            cfg.pop("tasks")
            env_cls = DOMAIN_REGISTRY[domain_name]
            self.domain_envs.append(env_cls(**cfg))

        # ----- 2. 构造扁平化的任务索引 -----
        # task_names[i]            : 第 i 个全局任务名
        # task_domain_index[i]     : 第 i 个任务所属的域下标
        # task_local_index[i]      : 第 i 个任务在其所属域内的局部下标
        self.task_names: list[str] = []
        self.task_domain_index: list[int] = []
        self.task_local_index: list[int] = []
        for domain_idx, domain_name in enumerate(self.domain_names):
            tasks = list(self.domain_configs[domain_name]["tasks"])
            for local_idx, task_name in enumerate(tasks):
                self.task_names.append(task_name)
                self.task_domain_index.append(domain_idx)
                self.task_local_index.append(local_idx)

        self.n_total_tasks: int = len(self.task_names)
        self.n_domains: int = len(self.domain_envs)

        # ----- 3. 计算对齐用的最大尺寸 -----
        self.domain_n_agents: list[int] = [
            len(env.possible_agents) for env in self.domain_envs
        ]
        self.max_n_agents: int = max(self.domain_n_agents)

        # 每个域中每个智能体的真实观测/动作维度
        self.domain_obs_dims: list[list[int]] = [
            [
                env.observation_space(agent).shape[0]
                for agent in env.possible_agents
            ]
            for env in self.domain_envs
        ]
        self.domain_action_dims: list[list[int]] = [
            [
                env.action_space(agent).shape[0]
                for agent in env.possible_agents
            ]
            for env in self.domain_envs
        ]

        max_raw_obs_dim = max(
            dim for dims in self.domain_obs_dims for dim in dims
        )
        self.max_action_dim: int = max(
            dim for dims in self.domain_action_dims for dim in dims
        )

        # 每个智能体的对外观测形状 = 任务 one-hot + 最大原始观测
        self.observation_dim: int = self.n_total_tasks + max_raw_obs_dim
        self.shared_state_dim: int = 1  # 与 m3w-marl 保持一致：占位

        # ----- 4. 对外暴露的统一空间 -----
        self.agent_names: list[str] = [
            f"agent_{i}" for i in range(self.max_n_agents)
        ]
        self.observation_space: list[Box] = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.observation_dim,),
                dtype=np.float32,
            )
            for _ in range(self.max_n_agents)
        ]
        self.shared_observation_space: list[Box] = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.shared_state_dim,),
                dtype=np.float32,
            )
            for _ in range(self.max_n_agents)
        ]
        self.action_space: list[Box] = [
            Box(
                low=-1.0,
                high=1.0,
                shape=(self.max_action_dim,),
                dtype=np.float32,
            )
            for _ in range(self.max_n_agents)
        ]

        # ----- 5. 当前激活任务 -----
        self.current_task_index: int = 0

    # ------------------------------------------------------------------
    # 标准 RL 接口
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[None]]:
        """
        复位当前激活任务对应的子环境。

        参数:
            seed: 随机种子。

        返回:
            ``(observations, shared_observations, available_actions)`` 三元组：

            * ``observations``        : 长度为 ``max_n_agents`` 的观测列表，
              每条均已加上任务 one-hot 并 padding 到 ``observation_dim``。
            * ``shared_observations`` : 占位的共享状态列表（每条形状 ``(1,)``）。
            * ``available_actions``   : 占位列表，元素均为 ``None``。
        """
        # 通知子环境切换到当前激活任务对应的局部任务
        self._sync_active_task()

        raw_obs_dict, _ = self.current_env.reset(seed=seed)

        observations = self._pad_observations(raw_obs_dict)
        shared_observations = [
            np.zeros(self.shared_state_dim, dtype=np.float32)
            for _ in range(self.max_n_agents)
        ]
        available_actions = [None] * self.max_n_agents
        return observations, shared_observations, available_actions

    def step(
        self,
        actions: list[np.ndarray],
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[bool],
        list[dict[str, Any]],
        list[None],
    ]:
        """
        在当前激活任务上执行一步动作。

        参数:
            actions: 长度为 ``max_n_agents`` 的动作列表，每条形状均为
                ``(max_action_dim,)``。多余的智能体与动作维度会被自动裁剪。

        返回:
            标准的多智能体六元组
            ``(observations, shared_observations, rewards, dones, infos,
            available_actions)``：

            * ``observations`` / ``shared_observations`` : 同 ``reset``。
            * ``rewards``        : 每个智能体一个 ``shape=(1,)`` 的数组，
              数值为当前任务的整形奖励。
            * ``dones``          : 每个智能体共享同一个 ``done`` 标志。
            * ``infos``          : 每个智能体一份的 info 字典，已注入
              ``task`` / ``task_index`` / ``bad_transition`` 字段。
            * ``available_actions`` : 占位列表，元素均为 ``None``。
        """
        cropped_actions = self._crop_actions(actions)
        action_dict = {
            agent: cropped_actions[i]
            for i, agent in enumerate(self.current_env.possible_agents)
        }

        raw_obs, raw_rewards, raw_terms, raw_truncs, raw_infos = (
            self.current_env.step(action_dict)
        )

        observations = self._pad_observations(raw_obs)
        shared_observations = [
            np.zeros(self.shared_state_dim, dtype=np.float32)
            for _ in range(self.max_n_agents)
        ]

        # 当前任务的标量奖励：所有智能体共享同一数值
        scalar_reward = float(next(iter(raw_rewards.values())))
        rewards = [
            np.array([scalar_reward], dtype=np.float32)
            for _ in range(self.max_n_agents)
        ]

        # 是否结束：任一智能体 terminate 或 truncate 均视为整局结束
        episode_done = any(raw_terms.values()) or any(raw_truncs.values())
        dones = [episode_done] * self.max_n_agents

        # info 注入任务元信息
        merged_info = {}
        for agent_info in raw_infos.values():
            merged_info.update(agent_info)
        merged_info["task"] = self.current_task
        merged_info["task_index"] = self.current_task_index
        merged_info["domain"] = self.current_domain_name
        merged_info["bad_transition"] = False
        infos = [dict(merged_info) for _ in range(self.max_n_agents)]

        available_actions = [None] * self.max_n_agents
        return (
            observations,
            shared_observations,
            rewards,
            dones,
            infos,
            available_actions,
        )

    def reset_task(
        self,
        task: int | str | None = None,
    ) -> int:
        """
        切换当前激活任务。

        参数:
            task: 目标任务，可以是全局任务下标（``int``）或任务名（``str``）。
                传 ``None`` 时随机抽取一个任务。

        返回:
            切换后的全局任务下标。

        异常:
            ValueError: 当 ``task`` 是字符串但不在 ``task_names`` 中
                或为越界整数时抛出。
        """
        if task is None:
            task_index = int(np.random.randint(self.n_total_tasks))
        elif isinstance(task, str):
            if task not in self.task_names:
                raise ValueError(
                    f"未知的任务名 {task!r}，可选: {self.task_names}。"
                )
            task_index = self.task_names.index(task)
        else:
            if not 0 <= task < self.n_total_tasks:
                raise ValueError(
                    f"任务下标 {task} 越界，应在 [0, {self.n_total_tasks})。"
                )
            task_index = int(task)

        self.current_task_index = task_index
        self._sync_active_task()
        return self.current_task_index

    def close(self) -> None:
        """关闭所有子环境，释放底层 MuJoCo 资源。"""
        for env in self.domain_envs:
            env.close()

    # ------------------------------------------------------------------
    # 辅助查询
    # ------------------------------------------------------------------

    @property
    def current_domain_index(self) -> int:
        """返回当前激活任务所属的域下标。"""
        return self.task_domain_index[self.current_task_index]

    @property
    def current_domain_name(self) -> str:
        """返回当前激活任务所属的域名称。"""
        return self.domain_names[self.current_domain_index]

    @property
    def current_env(self) -> ParallelEnv:
        """返回当前激活任务对应的子环境实例。"""
        return self.domain_envs[self.current_domain_index]

    @property
    def current_task(self) -> str:
        """返回当前激活任务的名称。"""
        return self.task_names[self.current_task_index]

    @property
    def current_local_task_index(self) -> int:
        """返回当前激活任务在其所属域内的局部下标。"""
        return self.task_local_index[self.current_task_index]

    def get_action_mask(self) -> np.ndarray:
        """
        构造动作有效性掩码。

        返回:
            形状为 ``(n_total_tasks, max_n_agents, max_action_dim)`` 的
            ``float32`` 数组，``1`` 表示该任务下该智能体的该动作维度有效，
            ``0`` 表示无效（被零填充）。
        """
        mask = np.zeros(
            (self.n_total_tasks, self.max_n_agents, self.max_action_dim),
            dtype=np.float32,
        )
        for global_idx in range(self.n_total_tasks):
            domain_idx = self.task_domain_index[global_idx]
            n_agents = self.domain_n_agents[domain_idx]
            for agent_idx in range(n_agents):
                action_dim = self.domain_action_dims[domain_idx][agent_idx]
                mask[global_idx, agent_idx, :action_dim] = 1.0
        return mask

    def get_task_names(self) -> list[str]:
        """返回 ``"<域名>_<任务名>"`` 形式的全部任务标签。"""
        return [
            f"{self.domain_names[self.task_domain_index[i]]}_{name}"
            for i, name in enumerate(self.task_names)
        ]

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _sync_active_task(self) -> None:
        """通知当前域的子环境切换到对应的局部任务。"""
        local_task_name = self.current_env.tasks[self.current_local_task_index]
        self.current_env.reset_task(local_task_name)

    def _pad_observations(
        self,
        raw_obs_dict: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """
        将子环境的原始观测字典对齐到统一形状。

        每个智能体的输出观测结构为::

            [任务 one-hot (n_total_tasks)] + [原始观测] + [零填充]

        若当前域智能体数少于 ``max_n_agents``，缺失位置追加全零观测。

        参数:
            raw_obs_dict: 子环境返回的智能体观测字典。

        返回:
            长度为 ``max_n_agents``、每条形状为 ``(observation_dim,)`` 的
            观测列表。
        """
        task_one_hot = np.zeros(self.n_total_tasks, dtype=np.float32)
        task_one_hot[self.current_task_index] = 1.0

        padded_observations: list[np.ndarray] = []
        for agent in self.current_env.possible_agents:
            raw_obs = np.asarray(raw_obs_dict[agent], dtype=np.float32)
            padding_size = (
                self.observation_dim - self.n_total_tasks - raw_obs.shape[0]
            )
            padded_observations.append(
                np.concatenate(
                    [
                        task_one_hot,
                        raw_obs,
                        np.zeros(padding_size, dtype=np.float32),
                    ],
                )
            )

        # 不足 max_n_agents 的位置补零
        while len(padded_observations) < self.max_n_agents:
            padded_observations.append(
                np.zeros(self.observation_dim, dtype=np.float32),
            )
        return padded_observations

    def _crop_actions(
        self,
        actions: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        把统一形状的动作裁剪到当前域真实需要的维度。

        参数:
            actions: 长度为 ``max_n_agents`` 的动作列表，每条形状
                ``(max_action_dim,)``。

        返回:
            长度等于当前域智能体数的动作列表，每条形状为该智能体的
            真实动作维度。
        """
        domain_idx = self.current_domain_index
        n_agents = self.domain_n_agents[domain_idx]
        return [
            np.asarray(actions[i], dtype=np.float32)[
                : self.domain_action_dims[domain_idx][i]
            ]
            for i in range(n_agents)
        ]
