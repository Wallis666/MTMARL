"""
将 PettingZoo ParallelEnv (单 agent 配置) 包装为 Gymnasium 单智能体环境。

适用于 ``HalfCheetahMultiTask(agent_conf=None)``：底层 MaMuJoCo 仍是
ParallelEnv 接口，但只有一个 agent，本 wrapper 把 dict 拆成裸 ndarray，
让 CleanRL 风格的 SAC 训练脚本可以直接使用。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import ParallelEnv


class ParallelToSingleAgent(gym.Env):
    """把单 agent 的 ParallelEnv 转成 ``gymnasium.Env``。"""

    metadata = {"render_modes": []}

    def __init__(self, parallel_env: ParallelEnv) -> None:
        if len(parallel_env.possible_agents) != 1:
            raise ValueError(
                f"ParallelToSingleAgent 仅支持单 agent，"
                f"当前 possible_agents={parallel_env.possible_agents}"
            )
        self._env = parallel_env
        self._agent: str = parallel_env.possible_agents[0]

        self.observation_space = parallel_env.observation_space(self._agent)
        self.action_space = parallel_env.action_space(self._agent)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, infos = self._env.reset(seed=seed, options=options)
        return obs[self._agent], infos.get(self._agent, {})

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, rewards, terms, truncs, infos = self._env.step(
            {self._agent: np.asarray(action, dtype=np.float32)},
        )
        return (
            obs[self._agent],
            float(rewards[self._agent]),
            bool(terms[self._agent]),
            bool(truncs[self._agent]),
            infos.get(self._agent, {}),
        )

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped_parallel(self) -> ParallelEnv:
        """返回底层 ParallelEnv，方便外部调用 ``reset_task`` 等多任务接口。"""
        return self._env
