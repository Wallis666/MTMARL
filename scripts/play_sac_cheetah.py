"""
加载 SAC checkpoint 并可视化 / 评估 HalfCheetahMultiTask。

用法::

    # 弹窗看着它跑
    python scripts/play_sac_cheetah.py \
        --ckpt runs/sac_cheetah_run_seed1_1775709085/ckpt/step_50000.pt \
        --task run --episodes 3 --render

    # 只跑数值评估，不开窗
    python scripts/play_sac_cheetah.py --ckpt ... --episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.algos.sac import SAC, SACConfig
from src.envs.mamujoco.tasks.cheetah import TASKS, HalfCheetahMultiTask
from src.wrappers.single_agent import ParallelToSingleAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--task", type=str, default="run", choices=list(TASKS))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--render", action="store_true",
                   help="开启 MuJoCo human 渲染窗口")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env_kwargs = {"render_mode": "human"} if args.render else {}
    parallel = HalfCheetahMultiTask(
        agent_conf=None,
        default_task=args.task,
        **env_kwargs,
    )
    env = ParallelToSingleAgent(parallel)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = SAC(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=env.action_space.low.astype(np.float32),
        action_high=env.action_space.high.astype(np.float32),
        device=args.device,
        config=SACConfig(),
    )
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    agent.load_state_dict(sd)
    agent.actor.eval()
    print(f"已加载 checkpoint: {args.ckpt}")

    returns: list[float] = []
    x_vels: list[float] = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_ret = 0.0
        ep_len = 0
        ep_xvel: list[float] = []
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            ep_len += 1
            if "x_velocity" in info:
                ep_xvel.append(float(info["x_velocity"]))
            done = term or trunc
        mean_xv = float(np.mean(ep_xvel)) if ep_xvel else float("nan")
        returns.append(ep_ret)
        x_vels.append(mean_xv)
        print(f"  ep {ep}: return={ep_ret:.2f} len={ep_len} mean_x_vel={mean_xv:.2f}")

    print(
        f"\n[{args.task}] {args.episodes} 回合: "
        f"return mean={np.mean(returns):.2f} std={np.std(returns):.2f} | "
        f"x_vel mean={np.nanmean(x_vels):.2f}"
    )
    env.close()


if __name__ == "__main__":
    main()
