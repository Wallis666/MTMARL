"""环境相关的工具函数。"""

import os
import random

import numpy as np
import torch

from baselines.envs.wrappers import (
    ShareDummyVecEnv,
    ShareSubprocVecEnv,
)


def check(value):
    """检查值是否为numpy数组，如果是则转换为torch张量。"""
    output = (
        torch.from_numpy(value)
        if isinstance(value, np.ndarray)
        else value
    )
    return output


def get_shape_from_obs_space(obs_space):
    """从观测空间中获取形状。

    参数:
        obs_space: 观测空间。
    返回:
        obs_shape: 观测形状。
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """从动作空间中获取形状。

    参数:
        act_space: 动作空间。
    返回:
        act_shape: 动作形状。
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def make_train_env(
    env_name,
    seed,
    n_threads,
    env_args,
):
    """创建训练环境。"""
    def get_env_fn(rank):
        def init_env():
            if env_name == "mamujoco":
                from baselines.envs.mamujoco.mamujoco import (
                    MaMuJoCoEnv,
                )

                env = MaMuJoCoEnv(
                    env_args=env_args,
                )
            else:
                print(
                    "不支持该环境: " + env_name
                )
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(n_threads)]
        )


def make_eval_env(
    env_name,
    seed,
    n_threads,
    env_args,
):
    """创建评估环境。"""
    # dexhands不支持同时运行多个实例
    if env_name == "dexhands":
        raise NotImplementedError

    def get_env_fn(rank):
        def init_env():
            if env_name == "mamujoco":
                from baselines.envs.mamujoco.mamujoco import (
                    MaMuJoCoEnv,
                )

                env = MaMuJoCoEnv(
                    env_args=env_args,
                )
            else:
                print(
                    "不支持该环境: " + env_name
                )
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv(
            [get_env_fn(i) for i in range(n_threads)]
        )


def make_render_env(
    env_name,
    seed,
    env_args,
):
    """创建渲染环境。"""
    # 手动调用render()函数
    manual_render = True
    # 手动扩展并行环境数量维度
    manual_expand_dims = True
    # 通过time.sleep()手动延迟渲染
    manual_delay = True
    # 并行环境数量
    env_num = 1
    if env_name == "mamujoco":
        from baselines.envs.mamujoco.mamujoco import (
            MaMuJoCoEnv,
        )

        env = MaMuJoCoEnv(
            env_args=env_args,
        )
        env.seed(seed * 60000)
    else:
        print("不支持该环境: " + env_name)
        raise NotImplementedError
    return (
        env,
        manual_render,
        manual_expand_dims,
        manual_delay,
        env_num,
    )


def set_seed(args):
    """设置程序的随机种子。"""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(
    env,
    env_args,
    envs,
):
    """获取环境中的智能体数量。"""
    if env == "mamujoco":
        return envs.n_agents
