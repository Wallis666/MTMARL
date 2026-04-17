"""Baselines 模型渲染脚本。

支持弹窗实时渲染和录制视频两种模式。

用法:
    # 弹窗渲染（默认，需要显示器）
    python -m scripts.render_baselines --algo mappo --env mamujoco \
        --model_dir baselines/runs/.../models

    # 录制视频（无头服务器也可用）
    python -m scripts.render_baselines --algo mappo --env mamujoco \
        --model_dir baselines/runs/.../models \
        --render_mode rgb_array --video_dir baselines/runs/videos

    # 指定回合数
    python -m scripts.render_baselines --algo mappo --env mamujoco \
        --model_dir baselines/runs/.../models --render_episodes 5
"""

import argparse
import os

# 在导入 mujoco 之前设置离屏渲染后端，
# 使 rgb_array 模式在无头服务器上也能工作。
# EGL 需要 GPU 驱动支持，OSMesa 为纯 CPU 软渲染备选。
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import imageio
import numpy as np

from baselines.runners import RUNNER_REGISTRY
from baselines.utils.config import (
    get_defaults_yaml_args,
    update_args,
)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=list(RUNNER_REGISTRY.keys()),
        help="算法名称",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mamujoco",
        help="环境名称",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="render",
        help="实验名称",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="训练好的模型目录路径",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="渲染模式: human 弹窗渲染, rgb_array 录制视频",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=10,
        help="渲染回合数",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="baselines/runs/videos",
        help="视频保存目录（仅 rgb_array 模式）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率（仅 rgb_array 模式）",
    )

    args, unparsed_args = parser.parse_known_args()

    # 加载 yaml 默认配置
    algo_args, env_args = get_defaults_yaml_args(
        args.algo, args.env
    )

    # 解析额外的 --key value 参数
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = dict(zip(keys, values))
    update_args(unparsed_dict, algo_args, env_args)

    # 强制设置渲染相关参数
    algo_args["render"]["use_render"] = True
    algo_args["render"]["render_episodes"] = (
        args.render_episodes
    )
    algo_args["train"]["model_dir"] = args.model_dir
    env_args["render_mode"] = args.render_mode

    args_dict = {
        "algo": args.algo,
        "env": args.env,
        "exp_name": args.exp_name,
    }

    runner = RUNNER_REGISTRY[args.algo](
        args_dict, algo_args, env_args
    )

    if args.render_mode == "human":
        runner.render()
    else:
        # rgb_array 模式: 逐帧收集并保存为视频
        os.makedirs(args.video_dir, exist_ok=True)
        task = env_args.get("task", "unknown")
        for ep in range(args.render_episodes):
            frames = collect_frames(runner)
            video_path = os.path.join(
                args.video_dir,
                f"{args.algo}_{task}_ep{ep}.mp4",
            )
            imageio.mimsave(
                video_path, frames, fps=args.fps
            )
            print(f"已保存视频: {video_path}")

    runner.close()


def collect_frames(runner):
    """收集一个回合的渲染帧。"""
    frames = []
    eval_obs, _, eval_available_actions = (
        runner.envs.reset()
    )
    eval_obs = np.expand_dims(
        np.array(eval_obs), axis=0
    )
    eval_available_actions = (
        np.expand_dims(
            np.array(eval_available_actions), axis=0
        )
        if eval_available_actions is not None
        else None
    )
    eval_rnn_states = np.zeros(
        (
            runner.env_num,
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_masks = np.ones(
        (runner.env_num, runner.num_agents, 1),
        dtype=np.float32,
    )

    from baselines.utils.trans import _t2n

    rewards = 0
    while True:
        eval_actions_collector = []
        for agent_id in range(runner.num_agents):
            eval_actions, temp_rnn_state = runner.actor[
                agent_id
            ].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions is not None
                else None,
                deterministic=True,
            )
            eval_rnn_states[:, agent_id] = _t2n(
                temp_rnn_state
            )
            eval_actions_collector.append(
                _t2n(eval_actions)
            )
        eval_actions = np.array(
            eval_actions_collector
        ).transpose(1, 0, 2)

        (
            eval_obs,
            _,
            eval_rewards,
            eval_dones,
            _,
            eval_available_actions,
        ) = runner.envs.step(eval_actions[0])

        rewards += eval_rewards[0][0]

        frame = runner.envs.render()
        if frame is not None:
            frames.append(frame)

        eval_obs = np.expand_dims(
            np.array(eval_obs), axis=0
        )
        eval_available_actions = (
            np.expand_dims(
                np.array(eval_available_actions),
                axis=0,
            )
            if eval_available_actions is not None
            else None
        )

        if eval_dones[0]:
            print(
                f"回合总奖励: {rewards}"
            )
            break

    return frames


def process(arg):
    """尝试将字符串参数转换为 Python 字面量。"""
    try:
        return eval(arg)
    except Exception:
        return arg


if __name__ == "__main__":
    main()
