"""
MTMARL 训练入口脚本。

读取一个 YAML 配置文件，依次构造 ``MultiTaskMaMuJoCo`` 向量化环境、
五大模型组件（encoder / dynamics / reward / actor / critic）、
``ReplayBuffer``、``WorldModelTrainer`` 与 ``Runner``，最后调用
``Runner.run()`` 启动训练主循环。

运行方式::

    python -m scripts.train --config configs/mamujoco/train_small.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from src.algos.runner import Runner
from src.algos.trainer import WorldModelTrainer
from src.buffers.replay_buffer import ReplayBuffer
from src.envs.mamujoco.multi_task import MultiTaskMaMuJoCo
from src.models.actor import SquashedGaussianActor
from src.models.critic import TwinQCritic
from src.models.dynamics import SoftMoEDynamics
from src.models.obs_encoder import ObsEncoder
from src.models.reward import SparseMoERewardModel
from src.utils.two_hot import TwoHotProcessor
from src.wrappers import ShareDummyVecEnv, ShareSubprocVecEnv


def _parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="MTMARL 训练入口脚本。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mamujoco/train_small.yaml",
        help="YAML 配置文件路径。",
    )
    return parser.parse_args()


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    """加载 YAML 配置文件为字典。"""
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _set_global_seed(
    seed: int,
) -> None:
    """统一设置 Python / NumPy / PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env_factory(
    domains_config: dict[str, Any],
) -> Callable[[], MultiTaskMaMuJoCo]:
    """构造一个无参环境工厂，便于多次创建独立的总线环境实例。"""

    def factory() -> MultiTaskMaMuJoCo:
        return MultiTaskMaMuJoCo(domains=domains_config)

    return factory


def _build_vec_env(
    domains_config: dict[str, Any],
    num_envs: int,
    use_subproc: bool,
):
    """根据配置构造单进程或多进程向量化环境。"""
    factory = _make_env_factory(domains_config)
    env_fns = [factory for _ in range(num_envs)]
    if use_subproc:
        return ShareSubprocVecEnv(env_fns)
    return ShareDummyVecEnv(env_fns)


def _build_models(
    sample_env: MultiTaskMaMuJoCo,
    model_config: dict[str, Any],
) -> tuple[
    ObsEncoder,
    SoftMoEDynamics,
    SparseMoERewardModel,
    SquashedGaussianActor,
    TwinQCritic,
]:
    """根据 ``sample_env`` 暴露的维度信息与模型超参构造五个模型。"""
    obs_dim = sample_env.observation_dim
    action_dim = sample_env.max_action_dim
    n_agents = sample_env.max_n_agents
    latent_dim = int(model_config["latent_dim"])
    simnorm_group_dim = int(model_config["simnorm_group_dim"])
    num_bins = int(model_config["num_bins"])

    encoder = ObsEncoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(model_config["encoder_hidden_dims"]),
        simnorm_group_dim=simnorm_group_dim,
    )
    dynamics = SoftMoEDynamics(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        n_experts=int(model_config["dynamics_n_experts"]),
        n_slots_per_expert=int(model_config["dynamics_n_slots_per_expert"]),
        hidden_dims=tuple(model_config["dynamics_hidden_dims"]),
        simnorm_group_dim=simnorm_group_dim,
    )
    reward_model = SparseMoERewardModel(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        n_experts=int(model_config["reward_n_experts"]),
        top_k=int(model_config["reward_top_k"]),
        num_bins=num_bins,
        n_attention_heads=int(model_config["reward_attention_heads"]),
        expert_ffn_hidden_dim=int(model_config["reward_ffn_hidden_dim"]),
        head_hidden_dim=int(model_config["reward_head_hidden_dim"]),
    )
    actor = SquashedGaussianActor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=tuple(model_config["actor_hidden_dims"]),
    )
    critic = TwinQCritic(
        n_agents=n_agents,
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dims=tuple(model_config["critic_hidden_dims"]),
        num_bins=num_bins,
    )
    return encoder, dynamics, reward_model, actor, critic


def main() -> None:
    """脚本入口。"""
    args = _parse_arguments()
    config = _load_yaml(Path(args.config))

    runtime_config = config.get("runtime", {})
    seed = int(runtime_config.get("seed", 0))
    device = str(runtime_config.get("device", "cpu"))
    _set_global_seed(seed)

    env_config = config["env"]
    train_config = config["train"]
    model_config = config["model"]
    algo_config = config["algo"]

    print("=" * 60)
    print("MTMARL 训练入口")
    print("=" * 60)
    print(f"  config : {args.config}")
    print(f"  device : {device}")
    print(f"  seed   : {seed}")

    # ------------------------------------------------------------------
    # 构造向量化环境
    # ------------------------------------------------------------------
    use_subproc = bool(env_config.get("use_subproc", False))
    train_env = _build_vec_env(
        domains_config=env_config["domains"],
        num_envs=int(env_config["num_train_envs"]),
        use_subproc=use_subproc,
    )
    eval_env = _build_vec_env(
        domains_config=env_config["domains"],
        num_envs=int(env_config["num_eval_envs"]),
        use_subproc=use_subproc,
    )

    # 用一个独立的临时实例读取维度信息（避免污染向量化环境状态）
    sample_env = MultiTaskMaMuJoCo(domains=env_config["domains"])
    obs_dim = sample_env.observation_dim
    shared_state_dim = sample_env.shared_state_dim
    action_dim = sample_env.max_action_dim
    n_agents = sample_env.max_n_agents
    sample_env.close()

    print(f"  obs_dim          : {obs_dim}")
    print(f"  shared_state_dim : {shared_state_dim}")
    print(f"  action_dim       : {action_dim}")
    print(f"  n_agents         : {n_agents}")

    # ------------------------------------------------------------------
    # 构造模型 / two-hot / buffer / trainer / runner
    # ------------------------------------------------------------------
    encoder, dynamics, reward_model, actor, critic = _build_models(
        sample_env=MultiTaskMaMuJoCo(domains=env_config["domains"]),
        model_config=model_config,
    )

    two_hot = TwoHotProcessor(
        num_bins=int(model_config["num_bins"]),
        vmin=float(model_config["reward_min"]),
        vmax=float(model_config["reward_max"]),
        device=device,
    )

    buffer = ReplayBuffer(
        buffer_size=int(train_config["buffer_size"]),
        num_envs=int(env_config["num_train_envs"]),
        n_agents=n_agents,
        obs_dim=obs_dim,
        shared_state_dim=shared_state_dim,
        action_dim=action_dim,
        n_step=int(algo_config["n_step"]),
        gamma=float(algo_config["gamma"]),
    )

    trainer = WorldModelTrainer(
        encoder=encoder,
        dynamics=dynamics,
        reward_model=reward_model,
        actor=actor,
        critic=critic,
        two_hot=two_hot,
        device=device,
        world_model_lr=float(algo_config["world_model_lr"]),
        actor_lr=float(algo_config["actor_lr"]),
        critic_lr=float(algo_config["critic_lr"]),
        gamma=float(algo_config["gamma"]),
        polyak=float(algo_config["polyak"]),
        alpha=float(algo_config["alpha"]),
        consistency_coef=float(algo_config["consistency_coef"]),
        reward_coef=float(algo_config["reward_coef"]),
        balance_coef=float(algo_config["balance_coef"]),
        rho=float(algo_config["rho"]),
        grad_clip_norm=float(algo_config["grad_clip_norm"]),
    )

    runner = Runner(
        train_env=train_env,
        eval_env=eval_env,
        buffer=buffer,
        trainer=trainer,
        num_env_steps=int(train_config["num_env_steps"]),
        warmup_steps=int(train_config["warmup_steps"]),
        update_every=int(train_config["update_every"]),
        updates_per_step=int(train_config["updates_per_step"]),
        batch_size=int(train_config["batch_size"]),
        horizon=int(train_config["horizon"]),
        eval_interval=int(train_config["eval_interval"]),
        eval_episodes=int(train_config["eval_episodes"]),
        log_interval=int(train_config["log_interval"]),
    )

    try:
        runner.run()
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
