"""
多智能体多任务训练 runner。

把 ``ShareVecEnv`` 包装的 ``MultiTaskMaMuJoCo`` 向量化环境、
``ReplayBuffer`` 与 :class:`WorldModelTrainer` 串成一个完整的
"采样 → 写入 → 更新 → 评估 → 日志" 主循环。设计目标是**简单可读**，
不引入分布式 / 多 logger / checkpoint 调度等复杂特性，先把流水线
打通；这些增强可在外层包一层而不必侵入本模块。

主循环高层结构::

    obs, shared_obs, _ = env.reset()
    for global_step in range(num_env_steps):
        if global_step < warmup_steps:
            actions = 随机均匀采样
        else:
            actions = trainer.select_actions(obs, stochastic=True)

        next_obs, next_shared, reward, done, infos, _ = env.step(actions)
        buffer.insert(obs, shared_obs, actions, reward, done, term,
                      next_obs, next_shared)

        if 训练条件满足:
            for _ in range(updates_per_step):
                trainer.update_world_model(buffer.sample_horizon(...))
                trainer.update_critic(buffer.sample(...))
                trainer.update_actor(buffer.sample(...))
                trainer.soft_update_target()

        if 评估条件满足:
            self.evaluate()

        obs, shared_obs = next_obs, next_shared

约定:

* 训练阶段每条 episode 结束后由向量化环境自动 ``reset``，runner
  仅负责累积每个并行环境的 episode 回报并在 done 时打印 / 存档。
* 评估阶段独立调用 ``vec_env.reset_task(task)`` 切到目标任务，跑
  指定数量的 episode，统计平均回报。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.algos.trainer import WorldModelTrainer
from src.buffers.replay_buffer import ReplayBuffer
from src.wrappers.base import ShareVecEnv


class Runner:
    """
    端到端训练 runner。

    持有训练所需的全部外部组件，自身不做模型构造，便于在外层灵活
    组合不同的 env / trainer / buffer。
    """

    def __init__(
        self,
        train_env: ShareVecEnv,
        eval_env: ShareVecEnv,
        buffer: ReplayBuffer,
        trainer: WorldModelTrainer,
        num_env_steps: int,
        warmup_steps: int = 5_000,
        update_every: int = 1,
        updates_per_step: int = 1,
        batch_size: int = 256,
        horizon: int = 3,
        eval_interval: int = 10_000,
        eval_episodes: int = 5,
        log_interval: int = 1_000,
    ) -> None:
        """
        参数:
            train_env: 训练用向量化环境。
            eval_env: 评估用向量化环境，推荐与训练环境相同任务集合
                但独立实例，避免相互污染。
            buffer: 经验回放缓冲区。
            trainer: 已构造好的 ``WorldModelTrainer``。
            num_env_steps: 总环境交互步数（按"时间步"计，不是按 transition 计）。
            warmup_steps: 起步阶段使用随机动作的步数，用于填充缓冲区。
            update_every: 每隔多少环境步触发一次更新。
            updates_per_step: 每次触发时连续做多少次梯度更新。
            batch_size: critic / actor 与世界模型更新使用的 batch 大小。
            horizon: 世界模型 ``sample_horizon`` 的序列长度。
            eval_interval: 每隔多少环境步评估一次。
            eval_episodes: 每个任务每次评估跑多少 episode。
            log_interval: 每隔多少环境步打印一次训练日志。
        """
        self.train_env = train_env
        self.eval_env = eval_env
        self.buffer = buffer
        self.trainer = trainer

        self.num_env_steps = num_env_steps
        self.warmup_steps = warmup_steps
        self.update_every = update_every
        self.updates_per_step = updates_per_step
        self.batch_size = batch_size
        self.horizon = horizon
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.log_interval = log_interval

        self.num_envs = train_env.num_envs
        self.n_agents = train_env.n_agents
        self.action_dim = trainer.actor.action_dim

        # 记录每个并行环境的当前 episode 回报与步数
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        # 已经完成的 episode 历史回报，用于做滑动平均
        self._completed_returns: list[float] = []
        self._completed_lengths: list[int] = []

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        """启动训练主循环。"""
        print("=" * 60)
        print("MTMARL 训练 runner 启动")
        print("=" * 60)
        print(f"  num_env_steps    : {self.num_env_steps}")
        print(f"  num_envs         : {self.num_envs}")
        print(f"  n_agents         : {self.n_agents}")
        print(f"  action_dim       : {self.action_dim}")
        print(f"  warmup_steps     : {self.warmup_steps}")
        print(f"  batch_size       : {self.batch_size}")
        print(f"  horizon          : {self.horizon}")
        print(f"  updates_per_step : {self.updates_per_step}")
        print()

        obs, shared_obs, _ = self.train_env.reset()
        start_time = time.time()
        latest_losses: dict[str, float] = {}

        for global_step in range(1, self.num_env_steps + 1):
            # 1) 选动作
            if global_step <= self.warmup_steps:
                actions = self._sample_random_actions()
            else:
                actions = self.trainer.select_actions(
                    obs,
                    stochastic=True,
                )

            # 2) 与环境交互
            (
                next_obs,
                next_shared_obs,
                rewards,
                dones,
                infos,
                _,
            ) = self.train_env.step(actions)
            terminations = self._extract_terminations(infos, dones)

            # 3) 写入缓冲区
            self.buffer.insert(
                obs=obs,
                shared_obs=shared_obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                terminations=terminations,
                next_obs=next_obs,
                next_shared_obs=next_shared_obs,
            )

            # 4) 累积回报并处理 episode 结束
            self._track_episode_returns(rewards, dones)

            obs = next_obs
            shared_obs = next_shared_obs

            # 5) 触发更新
            if (
                global_step > self.warmup_steps
                and global_step % self.update_every == 0
                and len(self.buffer) >= self.batch_size
            ):
                latest_losses = self._do_updates()

            # 6) 日志
            if global_step % self.log_interval == 0:
                self._log_progress(
                    global_step=global_step,
                    start_time=start_time,
                    latest_losses=latest_losses,
                )

            # 7) 评估
            if (
                self.eval_interval > 0
                and global_step % self.eval_interval == 0
            ):
                self.evaluate(global_step=global_step)

        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        global_step: int,
    ) -> dict[str, float]:
        """
        对所有任务依次评估，返回每个任务的平均 episode 回报。

        参数:
            global_step: 当前训练步数，仅用于日志打印。

        返回:
            ``{task_name: mean_return}`` 字典。
        """
        self.trainer.eval()
        try:
            task_names = self._collect_task_names()
            results: dict[str, float] = {}
            print(f"\n[评估] global_step={global_step}")
            for task_name in task_names:
                mean_return = self._evaluate_single_task(task_name)
                results[task_name] = mean_return
                print(f"  task={task_name:<16s} mean_return={mean_return:.3f}")
        finally:
            self.trainer.train()
        return results

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _sample_random_actions(self) -> np.ndarray:
        """采样形状 ``(num_envs, n_agents, action_dim)`` 的随机动作。"""
        return np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(self.num_envs, self.n_agents, self.action_dim),
        ).astype(np.float32)

    def _extract_terminations(
        self,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        从 ``infos`` 中提取真终止标志。

        若各 info 中含有 ``terminated`` / ``TimeLimit.truncated`` 字段
        则按"真终止 = terminated 且非截断"判断；否则退化为
        ``dones`` 自身（适用于任何环境）。
        """
        terminations = np.zeros(self.num_envs, dtype=np.bool_)
        for env_index, info in enumerate(infos):
            if isinstance(info, dict) and "terminated" in info:
                terminations[env_index] = bool(info["terminated"])
            else:
                terminations[env_index] = bool(np.asarray(dones).reshape(-1)[env_index])
        return terminations

    def _track_episode_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """累计每个并行环境的回合回报，并在 done 时记录到历史。"""
        rewards_flat = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
        dones_flat = np.asarray(dones, dtype=np.bool_).reshape(self.num_envs)
        self._episode_returns += rewards_flat
        self._episode_lengths += 1
        for env_index in range(self.num_envs):
            if dones_flat[env_index]:
                self._completed_returns.append(
                    float(self._episode_returns[env_index]),
                )
                self._completed_lengths.append(
                    int(self._episode_lengths[env_index]),
                )
                self._episode_returns[env_index] = 0.0
                self._episode_lengths[env_index] = 0

    def _do_updates(self) -> dict[str, float]:
        """连续做 ``updates_per_step`` 次梯度更新，返回最后一次的损失。"""
        latest: dict[str, float] = {}
        for _ in range(self.updates_per_step):
            horizon_batch = self.buffer.sample_horizon(
                batch_size=self.batch_size,
                horizon=self.horizon,
            )
            wm_losses = self.trainer.update_world_model(horizon_batch)

            transition_batch = self.buffer.sample(batch_size=self.batch_size)
            critic_losses = self.trainer.update_critic(transition_batch)
            actor_losses = self.trainer.update_actor(transition_batch)
            self.trainer.soft_update_target()

            latest = {**wm_losses, **critic_losses, **actor_losses}
        return latest

    def _log_progress(
        self,
        global_step: int,
        start_time: float,
        latest_losses: dict[str, float],
    ) -> None:
        """打印一次训练进度。"""
        elapsed = time.time() - start_time
        sps = global_step * self.num_envs / max(elapsed, 1e-6)
        recent_returns = self._completed_returns[-32:]
        mean_return = (
            float(np.mean(recent_returns)) if recent_returns else float("nan")
        )

        print(
            f"[step {global_step:>8d}] "
            f"buffer={len(self.buffer):>7d} "
            f"sps={sps:>7.1f} "
            f"ep_return(mean32)={mean_return:.3f}",
        )
        if latest_losses:
            loss_strs = [f"{key}={value:.4f}" for key, value in latest_losses.items()]
            print("  " + "  ".join(loss_strs))

    def _collect_task_names(self) -> list[str]:
        """从训练环境中提取一份去重的全局任务名列表。"""
        nested = self.train_env.get_task_names()
        seen: list[str] = []
        for per_env in nested:
            for name in per_env:
                if name not in seen:
                    seen.append(name)
        return seen

    def _evaluate_single_task(
        self,
        task_name: str,
    ) -> float:
        """对单个任务跑 ``eval_episodes`` 个回合并返回平均回报。"""
        self.eval_env.reset_task(task_name)
        obs, _, _ = self.eval_env.reset()

        per_env_return = np.zeros(self.eval_env.num_envs, dtype=np.float32)
        completed: list[float] = []

        target_episodes = self.eval_episodes
        max_safety_steps = 5_000  # 防止某些任务永远不结束
        for _ in range(max_safety_steps):
            actions = self.trainer.select_actions(obs, stochastic=False)
            obs, _, rewards, dones, _, _ = self.eval_env.step(actions)

            rewards_flat = np.asarray(rewards, dtype=np.float32).reshape(
                self.eval_env.num_envs,
            )
            dones_flat = np.asarray(dones, dtype=np.bool_).reshape(
                self.eval_env.num_envs,
            )
            per_env_return += rewards_flat
            for env_index in range(self.eval_env.num_envs):
                if dones_flat[env_index]:
                    completed.append(float(per_env_return[env_index]))
                    per_env_return[env_index] = 0.0
                    if len(completed) >= target_episodes:
                        break
            if len(completed) >= target_episodes:
                break

        return float(np.mean(completed)) if completed else 0.0
