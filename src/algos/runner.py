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
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from src.algos.trainer import WorldModelTrainer
from src.buffers.replay_buffer import ReplayBuffer
from src.utils.logger import TensorBoardLogger
from src.utils.video_recorder import write_video
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
        logger: TensorBoardLogger | None = None,
        ckpt_dir: str | Path | None = None,
        ckpt_interval: int = 50_000,
        render_env_factory: Callable[[], Any] | None = None,
        video_dir: str | Path | None = None,
        video_interval: int = 50_000,
        video_max_steps: int = 1000,
        video_fps: int = 30,
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
        self.logger = logger

        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else None
        if self.ckpt_dir is not None:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_interval = ckpt_interval

        self.render_env_factory = render_env_factory
        self.video_dir = Path(video_dir) if video_dir is not None else None
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_interval = video_interval
        self.video_max_steps = video_max_steps
        self.video_fps = video_fps

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

        progress_bar = tqdm(
            range(1, self.num_env_steps + 1),
            total=self.num_env_steps,
            desc="train",
            dynamic_ncols=True,
        )
        for global_step in progress_bar:
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
            # MultiTaskMaMuJoCo 给每个智能体复制同一份团队标量奖励 / done，
            # 这里压成 per-env 的形式喂给 buffer
            rewards_per_env = self._reduce_per_env(rewards)
            dones_per_env = self._reduce_per_env(dones).astype(bool)
            terminations = self._extract_terminations(infos, dones_per_env)

            # 3) 写入缓冲区
            self.buffer.insert(
                obs=obs,
                shared_obs=shared_obs,
                actions=actions,
                rewards=rewards_per_env,
                dones=dones_per_env,
                terminations=terminations,
                next_obs=next_obs,
                next_shared_obs=next_shared_obs,
            )

            # 4) 累积回报并处理 episode 结束
            self._track_episode_returns(rewards_per_env, dones_per_env)

            obs = next_obs
            shared_obs = next_shared_obs

            # 5) 触发更新
            if (
                global_step > self.warmup_steps
                and global_step % self.update_every == 0
                and len(self.buffer) >= self.batch_size
            ):
                latest_losses = self._do_updates()

            # 6) 进度条 + 日志
            recent_returns = self._completed_returns[-32:]
            mean_return = (
                float(np.mean(recent_returns)) if recent_returns else float("nan")
            )
            progress_bar.set_postfix(
                buffer=len(self.buffer),
                ep_ret=f"{mean_return:.2f}",
                wm=f"{latest_losses.get('wm/total', float('nan')):.3f}",
                cri=f"{latest_losses.get('critic/total', float('nan')):.3f}",
                act=f"{latest_losses.get('actor/loss', float('nan')):.3f}",
            )
            if global_step % self.log_interval == 0:
                self._log_progress(
                    global_step=global_step,
                    start_time=start_time,
                    latest_losses=latest_losses,
                )
                if self.logger is not None:
                    self.logger.log(
                        {
                            "train/ep_return": mean_return,
                            "train/buffer_size": float(len(self.buffer)),
                            **latest_losses,
                        },
                        step=global_step,
                    )

            # 7) 评估
            if (
                self.eval_interval > 0
                and global_step % self.eval_interval == 0
            ):
                self.evaluate(global_step=global_step)

            # 8) checkpoint
            if (
                self.ckpt_dir is not None
                and self.ckpt_interval > 0
                and global_step % self.ckpt_interval == 0
            ):
                self.save_checkpoint(global_step=global_step)

            # 9) 视频录制
            if (
                self.render_env_factory is not None
                and self.video_dir is not None
                and self.video_interval > 0
                and global_step % self.video_interval == 0
            ):
                self.record_videos(global_step=global_step)

        # 训练完成时强制保存一次
        if self.ckpt_dir is not None:
            self.save_checkpoint(global_step=self.num_env_steps, tag="final")

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
            ``{display_name: mean_return}`` 字典。
        """
        self.trainer.eval()
        try:
            display_names = self._collect_display_task_names()
            n_tasks = len(display_names)
            results: dict[str, float] = {}
            print(f"\n[评估] global_step={global_step}")
            for task_index in range(n_tasks):
                display_name = display_names[task_index]
                mean_return = self._evaluate_single_task(task_index)
                results[display_name] = mean_return
                print(
                    f"  task={display_name:<24s} "
                    f"mean_return={mean_return:.3f}",
                )
            if self.logger is not None:
                self.logger.log(
                    {f"eval/{task}": value for task, value in results.items()},
                    step=global_step,
                )
        finally:
            self.trainer.train()
        return results

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _reduce_per_env(
        self,
        per_agent_array: np.ndarray,
    ) -> np.ndarray:
        """
        把 ``(num_envs, n_agents, ...)`` 的 per-agent 数组压成
        ``(num_envs,)`` 的 per-env 标量。

        ``MultiTaskMaMuJoCo`` 给所有智能体复制同一份团队 reward / done，
        因此直接取第 0 个智能体的值即可。
        """
        array = np.asarray(per_agent_array)
        if array.ndim == 1:
            return array
        flat = array.reshape(self.num_envs, -1)
        return flat[:, 0]

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

    def _collect_display_task_names(self) -> list[str]:
        """
        从训练环境中提取一份用于展示 / TensorBoard 的全局任务名列表。

        ``ShareVecEnv.get_task_names()`` 返回的是 ``[per_env_list, ...]``
        的嵌套结构；因为所有 env 共享同一份任务集合，这里直接取第 0
        个 env 的列表即可。
        """
        nested = self.train_env.get_task_names()
        if isinstance(nested, list) and nested and isinstance(nested[0], list):
            return list(nested[0])
        return list(nested)

    def _evaluate_single_task(
        self,
        task_index: int,
    ) -> float:
        """对单个任务跑 ``eval_episodes`` 个回合并返回平均回报。"""
        # 用整数索引切换任务，避免依赖 ``reset_task`` 的字符串名约定
        self.eval_env.reset_task(task_index)
        obs, _, _ = self.eval_env.reset()

        per_env_return = np.zeros(self.eval_env.num_envs, dtype=np.float32)
        completed: list[float] = []

        target_episodes = self.eval_episodes
        max_safety_steps = 5_000  # 防止某些任务永远不结束
        for _ in range(max_safety_steps):
            actions = self.trainer.select_actions(obs, stochastic=False)
            obs, _, rewards, dones, _, _ = self.eval_env.step(actions)

            rewards_flat = np.asarray(rewards, dtype=np.float32)
            rewards_flat = rewards_flat.reshape(self.eval_env.num_envs, -1)[:, 0]
            dones_flat = np.asarray(dones, dtype=np.bool_)
            dones_flat = dones_flat.reshape(self.eval_env.num_envs, -1)[:, 0]
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

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        global_step: int,
        tag: str | None = None,
    ) -> Path:
        """
        把 trainer 的全部状态保存到 ``ckpt_dir``。

        参数:
            global_step: 当前训练步数，会写入文件名与 payload。
            tag: 可选的额外标签（如 ``"final"``）；为 ``None`` 时只用步数。

        返回:
            写入完成的 ``.pt`` 文件路径。

        异常:
            RuntimeError: 当未配置 ``ckpt_dir`` 时抛出。
        """
        if self.ckpt_dir is None:
            raise RuntimeError("未配置 ckpt_dir，无法保存 checkpoint。")
        filename = f"step_{global_step:08d}.pt" if tag is None else f"{tag}.pt"
        path = self.ckpt_dir / filename
        payload = {
            "global_step": global_step,
            "trainer": self.trainer.state_dict(),
        }
        torch.save(payload, path)

        # 同步覆盖 latest.pt，便于续训
        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(payload, latest_path)
        print(f"[ckpt] 已保存到 {path}")
        return path

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> int:
        """
        从 ``path`` 加载 trainer 状态并返回保存时的 ``global_step``。

        参数:
            path: ``.pt`` 文件路径，应来自 :meth:`save_checkpoint`。

        返回:
            恢复后的步数（仅供日志用，runner 主循环依然从 1 开始）。
        """
        payload = torch.load(path, map_location=self.trainer.device)
        self.trainer.load_state_dict(payload["trainer"])
        loaded_step = int(payload.get("global_step", 0))
        print(f"[ckpt] 已加载 {path}，原 global_step={loaded_step}")
        return loaded_step

    # ------------------------------------------------------------------
    # 视频录制
    # ------------------------------------------------------------------

    def record_videos(
        self,
        global_step: int,
    ) -> dict[str, Path]:
        """
        对每个任务录制一段确定性策略的可视化视频。

        每次会**新建**一个 ``render_mode='rgb_array'`` 的非向量化环境，
        跑完一个任务后立刻关闭，避免对训练 / 评估环境产生干扰。

        参数:
            global_step: 当前训练步数，会写入视频文件名。

        返回:
            ``{display_name: video_path}`` 字典。
        """
        if self.render_env_factory is None or self.video_dir is None:
            return {}

        results: dict[str, Path] = {}
        self.trainer.eval()
        try:
            render_env = self.render_env_factory()
            display_names = list(render_env.get_task_names())
            for task_index, display_name in enumerate(display_names):
                frames = self._rollout_video_frames(render_env, task_index)
                if not frames:
                    continue
                output_path = self.video_dir / (
                    f"step_{global_step:08d}_{display_name}.mp4"
                )
                write_video(frames, output_path, fps=self.video_fps)
                results[display_name] = output_path
                print(
                    f"[video] {display_name} -> {output_path} "
                    f"({len(frames)} frames)",
                )
            render_env.close()
        finally:
            self.trainer.train()
        return results

    def _rollout_video_frames(
        self,
        render_env: Any,
        task_index: int,
    ) -> list[np.ndarray]:
        """跑一个回合并收集每步的渲染帧（确定性策略）。"""
        render_env.reset_task(task_index)
        obs_list, _, _ = render_env.reset()
        frames: list[np.ndarray] = []
        for _ in range(self.video_max_steps):
            frame = render_env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

            obs_array = np.stack(obs_list)[None, ...]  # (1, n_agents, obs_dim)
            actions = self.trainer.select_actions(obs_array, stochastic=False)
            actions_per_agent = list(actions[0])

            obs_list, _, _, dones, _, _ = render_env.step(actions_per_agent)
            if bool(np.any(dones)):
                break
        return frames
