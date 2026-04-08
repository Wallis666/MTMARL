"""
多进程并行版向量化环境。

为每个持有的环境实例开启一个独立的子进程，主进程通过
``multiprocessing.Pipe`` 与子进程交换命令与结果。相比单进程版本，
能够把仿真步进和策略推理并行起来，显著提升 on-policy 训练吞吐。

子进程内部运行 :func:`_worker_loop`，按命令字典分发：

* ``step``           ：执行一步并自动 ``reset``。
* ``reset``          ：复位环境。
* ``reset_task``     ：切换激活任务。
* ``get_spaces``     ：返回观测/共享/动作空间元信息。
* ``get_action_mask``：返回当前任务的动作有效性掩码。
* ``get_task_names`` ：返回全局任务名列表。
* ``close``          ：清理并退出子进程。
"""

from __future__ import annotations

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Sequence

import numpy as np

from src.wrappers.base import ShareVecEnv
from src.wrappers.cloudpickle_wrapper import CloudpickleWrapper


def _is_episode_done(
    done_flags: Any,
) -> bool:
    """
    判断单个环境的 ``done`` 标志是否表示本回合已结束。

    参数:
        done_flags: 子环境 ``step`` 返回的 ``done`` 数组或标量。

    返回:
        ``True`` 表示该环境需要立即触发 ``reset``。
    """
    if isinstance(done_flags, (bool, np.bool_)):
        return bool(done_flags)
    return bool(np.all(np.asarray(done_flags)))


def _worker_loop(
    worker_remote: Connection,
    parent_remote: Connection,
    env_factory: CloudpickleWrapper,
) -> None:
    """
    子进程主循环。

    参数:
        worker_remote: 子进程一侧的管道端，用于和主进程通信。
        parent_remote: 主进程一侧的管道端，子进程持有后立即关闭。
        env_factory: 通过 ``CloudpickleWrapper`` 包装的环境工厂函数。
    """
    parent_remote.close()
    env = env_factory.wrapped_object()

    try:
        while True:
            command, payload = worker_remote.recv()
            match command:
                case "step":
                    (
                        obs,
                        shared_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                    ) = env.step(payload)
                    if _is_episode_done(dones):
                        obs, shared_obs, available_actions = env.reset()
                    worker_remote.send(
                        (
                            obs,
                            shared_obs,
                            rewards,
                            dones,
                            infos,
                            available_actions,
                        ),
                    )

                case "reset":
                    obs, shared_obs, available_actions = env.reset()
                    worker_remote.send((obs, shared_obs, available_actions))

                case "reset_task":
                    worker_remote.send(env.reset_task(payload))

                case "get_spaces":
                    worker_remote.send(
                        (
                            env.observation_space,
                            env.shared_observation_space,
                            env.action_space,
                        ),
                    )

                case "get_action_mask":
                    worker_remote.send(env.get_action_mask())

                case "get_task_names":
                    worker_remote.send(env.get_task_names())

                case "close":
                    worker_remote.send(None)
                    break

                case _:
                    raise NotImplementedError(
                        f"未知的子进程命令: {command!r}。"
                    )
    finally:
        env.close()
        worker_remote.close()


class ShareSubprocVecEnv(ShareVecEnv):
    """
    多进程并行版向量化环境。

    每个子进程独占一个底层环境实例并独立运行 ``_worker_loop``，主进程
    通过管道异步分发动作、回收结果并堆叠成 batch。
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
        self._waiting_for_step: bool = False
        self._closed: bool = False

        num_envs = len(env_fns)
        self.parent_remotes: list[Connection] = []
        self.worker_remotes: list[Connection] = []
        for _ in range(num_envs):
            parent_end, worker_end = Pipe()
            self.parent_remotes.append(parent_end)
            self.worker_remotes.append(worker_end)

        self.processes: list[Process] = []
        for parent_end, worker_end, env_fn in zip(
            self.parent_remotes,
            self.worker_remotes,
            env_fns,
            strict=True,
        ):
            process = Process(
                target=_worker_loop,
                args=(worker_end, parent_end, CloudpickleWrapper(env_fn)),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

        # 子进程已持有 worker_end，主进程需要释放自己这一份
        for worker_end in self.worker_remotes:
            worker_end.close()

        # 通过任意子进程读取空间元信息
        self.parent_remotes[0].send(("get_spaces", None))
        observation_space, shared_observation_space, action_space = (
            self.parent_remotes[0].recv()
        )
        self.n_agents: int = len(observation_space)

        super().__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            shared_observation_space=shared_observation_space,
            action_space=action_space,
        )

    # ------------------------------------------------------------------
    # ShareVecEnv 接口实现
    # ------------------------------------------------------------------

    def reset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """向所有子进程广播 ``reset`` 命令并堆叠返回结果。"""
        for remote in self.parent_remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.parent_remotes]
        observations, shared_observations, available_actions = zip(
            *results,
            strict=True,
        )

        return (
            np.stack([np.asarray(item) for item in observations]),
            np.stack([np.asarray(item) for item in shared_observations]),
            self._stack_available_actions(available_actions),
        )

    def step_async(
        self,
        actions: np.ndarray,
    ) -> None:
        """异步把每个环境对应的动作发送给各自的子进程。"""
        for remote, action in zip(self.parent_remotes, actions, strict=True):
            remote.send(("step", action))
        self._waiting_for_step = True

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
        """阻塞等待所有子进程返回，并把结果堆叠成 batch。"""
        results = [remote.recv() for remote in self.parent_remotes]
        self._waiting_for_step = False
        (
            observations,
            shared_observations,
            rewards,
            dones,
            infos,
            available_actions,
        ) = zip(*results, strict=True)

        return (
            np.stack([np.asarray(item) for item in observations]),
            np.stack([np.asarray(item) for item in shared_observations]),
            np.stack([np.asarray(item) for item in rewards]),
            np.stack([np.asarray(item) for item in dones]),
            list(infos),
            self._stack_available_actions(available_actions),
        )

    def reset_task(
        self,
        task: int | str | None,
    ) -> np.ndarray:
        """向所有子进程广播相同的任务切换命令。"""
        for remote in self.parent_remotes:
            remote.send(("reset_task", task))
        return np.stack([remote.recv() for remote in self.parent_remotes])

    def close_extras(self) -> None:
        """通知所有子进程退出，并 join 回收进程资源。"""
        if self._closed:
            return

        if self._waiting_for_step:
            for remote in self.parent_remotes:
                remote.recv()
            self._waiting_for_step = False

        for remote in self.parent_remotes:
            remote.send(("close", None))
        for remote in self.parent_remotes:
            try:
                remote.recv()
            except EOFError:
                pass
            remote.close()
        for process in self.processes:
            process.join()

        self._closed = True

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def reset_task_per_env(
        self,
        task_indices: Sequence[int],
    ) -> np.ndarray:
        """
        为每个并行环境分别指定不同的目标任务。

        参数:
            task_indices: 长度等于 ``num_envs`` 的任务索引序列。

        返回:
            形状 ``(num_envs,)`` 的实际切换后任务索引数组。

        异常:
            ValueError: 当 ``task_indices`` 长度与 ``num_envs`` 不一致时抛出。
        """
        if len(task_indices) != self.num_envs:
            raise ValueError(
                f"`task_indices` 长度 {len(task_indices)} 与 num_envs "
                f"{self.num_envs} 不一致。"
            )
        for remote, task_index in zip(
            self.parent_remotes,
            task_indices,
            strict=True,
        ):
            remote.send(("reset_task", task_index))
        return np.stack([remote.recv() for remote in self.parent_remotes])

    def get_action_mask(self) -> np.ndarray:
        """从所有子进程收集动作掩码并堆叠。"""
        for remote in self.parent_remotes:
            remote.send(("get_action_mask", None))
        return np.stack([remote.recv() for remote in self.parent_remotes])

    def get_task_names(self) -> list[list[str]]:
        """从所有子进程收集全局任务名列表。"""
        for remote in self.parent_remotes:
            remote.send(("get_task_names", None))
        return [remote.recv() for remote in self.parent_remotes]

    @staticmethod
    def _stack_available_actions(
        available_actions: Sequence[Any],
    ) -> np.ndarray | None:
        """若所有 ``available_actions`` 均为 ``None`` 则返回 ``None``。"""
        if all(item is None for item in available_actions):
            return None
        return np.stack([np.asarray(item) for item in available_actions])
