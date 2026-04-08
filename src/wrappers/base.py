"""
共享观测向量化环境抽象基类。

参考 OpenAI Baselines 的 ``VecEnv`` 与 m3w-marl 的 ``ShareVecEnv``，
针对 ``MultiTaskMaMuJoCo`` 这类 **多智能体 + 共享中心化状态 + 多任务**
的场景进行简化与现代化。

子类只需实现 :meth:`reset` / :meth:`step_async` / :meth:`step_wait`
与 :meth:`reset_task`，即可获得 ``step`` / ``close`` 等通用行为。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from gymnasium.spaces import Space


class ShareVecEnv(ABC):
    """
    多智能体共享观测向量化环境抽象基类。

    所有具体子类都应满足如下约定（与 ``MultiTaskMaMuJoCo`` 对齐）：

    * 每次 ``reset`` 返回三元组
      ``(observations, shared_observations, available_actions)``，
      形状均带有最外层的 ``num_envs`` 维度。
    * 每次 ``step`` 返回六元组
      ``(observations, shared_observations, rewards, dones, infos,
      available_actions)``。
    * ``reset_task`` 用于切换激活任务，可接受单个任务索引 / 任务名 /
      ``None`` （随机），返回切换后的任务索引。
    """

    closed: bool = False

    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        num_envs: int,
        observation_space: Sequence[Space],
        shared_observation_space: Sequence[Space],
        action_space: Sequence[Space],
    ) -> None:
        """
        参数:
            num_envs: 并行环境实例的数量。
            observation_space: 每个智能体的观测空间列表。
            shared_observation_space: 每个智能体对应的共享状态空间列表。
            action_space: 每个智能体的动作空间列表。
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.shared_observation_space = shared_observation_space
        self.action_space = action_space

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        复位所有并行环境。

        返回:
            ``(observations, shared_observations, available_actions)``，
            其中前两项形状的最外层维度为 ``num_envs``。
        """

    @abstractmethod
    def step_async(
        self,
        actions: np.ndarray,
    ) -> None:
        """
        异步发送动作。

        参数:
            actions: 形状 ``(num_envs, n_agents, action_dim)`` 的动作数组。
        """

    @abstractmethod
    def step_wait(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        np.ndarray | None,
    ]:
        """
        等待并收集所有并行环境的 step 结果。

        返回:
            ``(observations, shared_observations, rewards, dones, infos,
            available_actions)``。
        """

    @abstractmethod
    def reset_task(
        self,
        task: int | str | None,
    ) -> np.ndarray:
        """
        切换所有并行环境的激活任务。

        参数:
            task: 目标任务索引 / 任务名 / ``None`` （由各子环境随机选择）。

        返回:
            形状 ``(num_envs,)`` 的数组，表示每个环境实际切换到的任务索引。
        """

    @abstractmethod
    def close_extras(self) -> None:
        """子类钩子：执行额外的资源释放逻辑。"""

    # ------------------------------------------------------------------
    # 通用方法
    # ------------------------------------------------------------------

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        np.ndarray | None,
    ]:
        """
        同步执行一步：先 ``step_async`` 再 ``step_wait``。

        参数:
            actions: 形状 ``(num_envs, n_agents, action_dim)`` 的动作数组。

        返回:
            与 :meth:`step_wait` 相同的六元组。
        """
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        """关闭向量化环境，幂等。"""
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    @property
    def unwrapped(self) -> "ShareVecEnv":
        """返回未被进一步包装的底层向量化环境（占位）。"""
        return self
