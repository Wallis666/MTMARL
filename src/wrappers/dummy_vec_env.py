"""
单进程顺序版向量化环境。

将多个环境实例放在主进程中顺序执行，接口与
:class:`ShareSubprocVecEnv` 完全一致，便于在调试阶段直接打断点
查看 traceback。生产训练场景请改用多进程版本以获得加速。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from src.wrappers.base import ShareVecEnv


class ShareDummyVecEnv(ShareVecEnv):
    """
    单进程顺序版向量化环境。

    每次 ``step`` 会按顺序遍历持有的环境实例，逐个调用其 ``step``，
    再把结果在最外层 ``num_envs`` 维度上 ``np.stack`` 拼接。
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
    ) -> None:
        """
        参数:
            env_fns: 一组无参可调用的环境工厂函数，每个调用应返回一个
                兼容 ``MultiTaskMaMuJoCo`` 接口的环境实例。
        """
        self.envs: list[Any] = [fn() for fn in env_fns]
        first_env = self.envs[0]

        self.n_agents: int = first_env.max_n_agents
        self._pending_actions: np.ndarray | None = None

        super().__init__(
            num_envs=len(self.envs),
            observation_space=first_env.observation_space,
            shared_observation_space=first_env.shared_observation_space,
            action_space=first_env.action_space,
        )

    # ------------------------------------------------------------------
    # ShareVecEnv 接口实现
    # ------------------------------------------------------------------

    def reset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """复位所有持有的环境并堆叠返回。"""
        results = [env.reset() for env in self.envs]
        observations, shared_observations, available_actions = zip(
            *results,
            strict=True,
        )

        stacked_obs = np.stack([np.asarray(item) for item in observations])
        stacked_shared = np.stack(
            [np.asarray(item) for item in shared_observations],
        )
        stacked_avail = self._stack_available_actions(available_actions)
        return stacked_obs, stacked_shared, stacked_avail

    def step_async(
        self,
        actions: np.ndarray,
    ) -> None:
        """记录待执行动作，由 ``step_wait`` 真正执行。"""
        self._pending_actions = actions

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
        顺序执行所有环境的一步动作，并对 ``done`` 的环境自动 ``reset``。
        """
        if self._pending_actions is None:
            raise RuntimeError("调用 step_wait 前必须先调用 step_async。")

        results = [
            env.step(action)
            for env, action in zip(self.envs, self._pending_actions, strict=True)
        ]
        self._pending_actions = None

        (
            observations,
            shared_observations,
            rewards,
            dones,
            infos,
            available_actions,
        ) = zip(*results, strict=True)

        observations = [np.asarray(item) for item in observations]
        shared_observations = [np.asarray(item) for item in shared_observations]
        rewards = [np.asarray(item) for item in rewards]
        dones = [np.asarray(item) for item in dones]

        # 自动 reset：若该环境本回合已结束，则立即复位以保持步进连续
        for env_idx, done_flags in enumerate(dones):
            if self._is_episode_done(done_flags):
                reset_obs, reset_shared, reset_avail = self.envs[env_idx].reset()
                observations[env_idx] = np.asarray(reset_obs)
                shared_observations[env_idx] = np.asarray(reset_shared)
                if reset_avail is not None:
                    available_actions = list(available_actions)
                    available_actions[env_idx] = reset_avail

        stacked_obs = np.stack(observations)
        stacked_shared = np.stack(shared_observations)
        stacked_rewards = np.stack(rewards)
        stacked_dones = np.stack(dones)
        stacked_avail = self._stack_available_actions(available_actions)

        return (
            stacked_obs,
            stacked_shared,
            stacked_rewards,
            stacked_dones,
            list(infos),
            stacked_avail,
        )

    def reset_task(
        self,
        task: int | str | None,
    ) -> np.ndarray:
        """对每个持有的环境广播相同的任务切换命令。"""
        return np.stack([env.reset_task(task) for env in self.envs])

    def close_extras(self) -> None:
        """关闭所有持有的环境。"""
        for env in self.envs:
            env.close()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """收集所有环境的动作掩码并堆叠返回。"""
        return np.stack([env.get_action_mask() for env in self.envs])

    def get_task_names(self) -> list[list[str]]:
        """收集所有环境的全局任务名列表。"""
        return [env.get_task_names() for env in self.envs]

    @staticmethod
    def _is_episode_done(
        done_flags: np.ndarray,
    ) -> bool:
        """
        判断单个环境的回合是否已结束。

        参数:
            done_flags: 该环境返回的 ``done`` 数组或标量。

        返回:
            ``True`` 表示需要触发自动 reset。
        """
        if isinstance(done_flags, (bool, np.bool_)):
            return bool(done_flags)
        return bool(np.all(done_flags))

    @staticmethod
    def _stack_available_actions(
        available_actions: Sequence[Any],
    ) -> np.ndarray | None:
        """
        将各环境的 ``available_actions`` 列表堆叠为单个数组。

        若所有元素均为 ``None``，返回 ``None``，避免后续算法出现
        ``np.stack([None, None])`` 的奇怪结果。
        """
        if all(item is None for item in available_actions):
            return None
        return np.stack([np.asarray(item) for item in available_actions])
