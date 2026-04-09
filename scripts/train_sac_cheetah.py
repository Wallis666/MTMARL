"""
单智能体 SAC 在 HalfCheetahMultiTask 上的训练脚本（CleanRL 风格）。

用法::

    python scripts/train_sac_cheetah.py --task run --total-steps 1000000

约定:
  * ``agent_conf=None`` → 单 agent 配置；
  * 任务通过 ``--task`` 指定，对应 ``cheetah.TASKS``；
  * 训练日志写入 ``runs/<run_name>/`` 下的 TensorBoard。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.algos.sac import SAC, SACConfig
from src.buffers.replay import ReplayBuffer
from src.envs.mamujoco.tasks.cheetah import TASKS, HalfCheetahMultiTask
from src.wrappers.single_agent import ParallelToSingleAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="run", choices=list(TASKS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--start-steps", type=int, default=5_000,
                   help="先用随机策略采集多少步再开始更新")
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def make_env(task: str, seed: int) -> ParallelToSingleAgent:
    parallel = HalfCheetahMultiTask(agent_conf=None, default_task=task)
    env = ParallelToSingleAgent(parallel)
    env.reset(seed=seed)
    return env


def evaluate(agent: SAC, task: str, episodes: int, seed: int) -> float:
    env = make_env(task, seed + 9999)
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 9999 + ep)
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def main() -> None:
    args = parse_args()

    run_name = args.run_name or f"sac_cheetah_{args.task}_seed{args.seed}_{int(time.time())}"
    log_dir = Path("runs") / run_name
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir.as_posix())
    writer.add_text("args", str(vars(args)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.task, args.seed)
    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    agent = SAC(
        obs_dim=int(np.prod(obs_shape)),
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        device=args.device,
        config=SACConfig(),
    )
    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=args.device,
    )

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_len = 0
    ep_count = 0
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        # 仅 terminated（环境真终止）算 done；truncated 视为 bootstrap
        buffer.add(obs, action, reward, next_obs, terminated)

        obs = next_obs
        ep_ret += reward
        ep_len += 1

        if terminated or truncated:
            ep_count += 1
            writer.add_scalar("rollout/ep_return", ep_ret, step)
            writer.add_scalar("rollout/ep_length", ep_len, step)
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

        # 更新
        if step >= args.start_steps and step % args.update_every == 0:
            batch = buffer.sample(args.batch_size)
            info_u = agent.update(batch)
            if step % 1000 == 0:
                for k, v in info_u.items():
                    writer.add_scalar(f"train/{k}", v, step)
                sps = step / (time.time() - t0)
                writer.add_scalar("train/sps", sps, step)
                print(f"[{step}] sps={sps:.1f} ep={ep_count} "
                      f"q_loss={info_u['q_loss']:.3f} "
                      f"actor_loss={info_u['actor_loss']:.3f} "
                      f"alpha={info_u['alpha']:.3f}")

        # 评估
        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, args.task, args.eval_episodes, args.seed)
            writer.add_scalar("eval/ep_return", eval_ret, step)
            print(f"[{step}] eval_return={eval_ret:.2f}")

        # 存档
        if step % args.save_every == 0:
            torch.save(agent.state_dict(), ckpt_dir / f"step_{step}.pt")

    torch.save(agent.state_dict(), ckpt_dir / "final.pt")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
