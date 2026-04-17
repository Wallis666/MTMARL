"""离线策略算法的基础运行器模块。"""

import os
import time

import numpy as np
import setproctitle
import torch
from torch.distributions import Categorical

from baselines.algos.actors import ALGO_REGISTRY
from baselines.algos.critics import CRITIC_REGISTRY
from baselines.common.buffers.off_policy_buffer_ep import (
    OffPolicyBufferEP,
)
from baselines.common.buffers.off_policy_buffer_fp import (
    OffPolicyBufferFP,
)
from baselines.common.value_norm import ValueNorm
from baselines.utils.config import (
    get_task_name,
    init_dir,
    save_config,
)
from baselines.utils.env import (
    get_num_agents,
    make_eval_env,
    make_render_env,
    make_train_env,
    set_seed,
)
from baselines.utils.model import init_device
from baselines.utils.trans import _t2n


class OffPolicyBaseRunner:
    """离线策略算法的基础运行器。"""

    def __init__(
        self,
        args,
        algo_args,
        env_args,
    ):
        """初始化 OffPolicyBaseRunner 类。

        参数:
            args: 由 argparse 解析的命令行参数。
                包含三个键: algo, env, exp_name。
            algo_args: 与算法相关的参数，从配置文件加载并
                用未解析的命令行参数更新。
            env_args: 与环境相关的参数，从配置文件加载并
                用未解析的命令行参数更新。
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        if "policy_freq" in self.algo_args["algo"]:
            self.policy_freq = (
                self.algo_args["algo"]["policy_freq"]
            )
        else:
            self.policy_freq = 1

        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args["render"]["use_render"]:
            (
                self.run_dir,
                self.log_dir,
                self.save_dir,
                self.writter,
            ) = init_dir(
                args["env"],
                env_args,
                args["algo"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(
                args, algo_args, env_args, self.run_dir
            )
            self.log_file = open(
                os.path.join(self.run_dir, "progress.txt"),
                "w",
                encoding="utf-8",
            )
        setproctitle.setproctitle(
            str(env_args.get("scenario", args["env"]))
            + "-"
            + str(args["algo"])
            + "-"
            + str(env_args.get("task", ""))
        )

        # 环境
        if self.algo_args["render"]["use_render"]:
            # 创建用于渲染的环境
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(
                args["env"],
                algo_args["seed"]["seed"],
                env_args,
            )
        else:
            # 创建用于训练和评估的环境
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"][
                        "n_eval_rollout_threads"
                    ],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(
            args["env"], env_args, self.envs
        )
        self.agent_deaths = np.zeros(
            (
                self.algo_args["train"]["n_rollout_threads"],
                self.num_agents,
                1,
            )
        )

        self.action_spaces = self.envs.action_space
        for agent_id in range(self.num_agents):
            self.action_spaces[agent_id].seed(
                algo_args["seed"]["seed"] + agent_id + 1
            )

        print(
            "share_observation_space: ",
            self.envs.share_observation_space,
        )
        print(
            "observation_space: ",
            self.envs.observation_space,
        )
        print(
            "action_space: ",
            self.envs.action_space,
        )

        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {
                    **algo_args["model"],
                    **algo_args["algo"],
                },
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), (
                    "智能体具有异构观测空间，"
                    "参数共享不适用。"
                )
                assert (
                    self.envs.action_space[agent_id]
                    == self.envs.action_space[0]
                ), (
                    "智能体具有异构动作空间，"
                    "参数共享不适用。"
                )
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {
                        **algo_args["model"],
                        **algo_args["algo"],
                    },
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        if not self.algo_args["render"]["use_render"]:
            self.critic = CRITIC_REGISTRY[args["algo"]](
                {
                    **algo_args["train"],
                    **algo_args["model"],
                    **algo_args["algo"],
                },
                self.envs.share_observation_space[0],
                self.envs.action_space,
                self.num_agents,
                self.state_type,
                device=self.device,
            )
            if self.state_type == "EP":
                self.buffer = OffPolicyBufferEP(
                    {
                        **algo_args["train"],
                        **algo_args["model"],
                        **algo_args["algo"],
                    },
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            elif self.state_type == "FP":
                self.buffer = OffPolicyBufferFP(
                    {
                        **algo_args["train"],
                        **algo_args["model"],
                        **algo_args["algo"],
                    },
                    self.envs.share_observation_space[0],
                    self.num_agents,
                    self.envs.observation_space,
                    self.envs.action_space,
                )
            else:
                raise NotImplementedError

        if (
            "use_valuenorm"
            in self.algo_args["train"].keys()
            and self.algo_args["train"]["use_valuenorm"]
        ):
            self.value_normalizer = ValueNorm(
                1, device=self.device
            )
        else:
            self.value_normalizer = None

        if self.algo_args["train"]["model_dir"] is not None:
            self.restore()

        self.total_it = 0  # 总迭代次数

        if (
            "auto_alpha"
            in self.algo_args["algo"].keys()
            and self.algo_args["algo"]["auto_alpha"]
        ):
            self.target_entropy = []
            for agent_id in range(self.num_agents):
                if (
                    self.envs.action_space[
                        agent_id
                    ].__class__.__name__
                    == "Box"
                ):
                    # 微分熵可以为负
                    self.target_entropy.append(
                        -np.prod(
                            self.envs.action_space[
                                agent_id
                            ].shape
                        )
                    )
                else:
                    # 离散熵总为正，因此将最大可能熵
                    # 设为目标熵
                    self.target_entropy.append(
                        -0.98
                        * np.log(
                            1.0
                            / np.prod(
                                self.envs.action_space[
                                    agent_id
                                ].shape
                            )
                        )
                    )
            self.log_alpha = []
            self.alpha_optimizer = []
            self.alpha = []
            for agent_id in range(self.num_agents):
                _log_alpha = torch.zeros(
                    1,
                    requires_grad=True,
                    device=self.device,
                )
                self.log_alpha.append(_log_alpha)
                self.alpha_optimizer.append(
                    torch.optim.Adam(
                        [_log_alpha],
                        lr=self.algo_args["algo"][
                            "alpha_lr"
                        ],
                    )
                )
                self.alpha.append(
                    torch.exp(_log_alpha.detach())
                )
        elif "alpha" in self.algo_args["algo"].keys():
            self.alpha = (
                [self.algo_args["algo"]["alpha"]]
                * self.num_agents
            )

    def run(self):
        """运行训练（或渲染）流程。"""
        if self.algo_args["render"]["use_render"]:
            # 渲染模式，不进行训练
            self.render()
            return
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        # 预热
        print("start warmup")
        obs, share_obs, available_actions = self.warmup()
        print("finish warmup, start training")
        # 训练和评估
        steps = (
            self.algo_args["train"]["num_env_steps"]
            // self.algo_args["train"][
                "n_rollout_threads"
            ]
        )
        # 每次训练的更新次数
        update_num = int(
            self.algo_args["train"]["update_per_train"]
            * self.algo_args["train"]["train_interval"]
        )
        for step in range(1, steps + 1):
            actions = self.get_actions(
                obs,
                available_actions=available_actions,
                add_random=True,
            )
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            # rewards: (n_threads, n_agents, 1)
            # dones: (n_threads, n_agents)
            # available_actions:
            #   (n_threads, ) of None 或
            #   (n_threads, n_agents, action_number)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = (
                new_available_actions.copy()
            )
            aa_shape = np.array(available_actions).shape
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(aa_shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(
                    1, 0, 2
                )
                if len(aa_shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
            train_interval = (
                self.algo_args["train"]["train_interval"]
            )
            if step % train_interval == 0:
                if self.algo_args["train"][
                    "use_linear_lr_decay"
                ]:
                    if self.share_param:
                        self.actor[0].lr_decay(
                            step, steps
                        )
                    else:
                        for agent_id in range(
                            self.num_agents
                        ):
                            self.actor[
                                agent_id
                            ].lr_decay(step, steps)
                    self.critic.lr_decay(step, steps)
                for _ in range(update_num):
                    self.train()
            eval_interval = (
                self.algo_args["train"]["eval_interval"]
            )
            if step % eval_interval == 0:
                n_threads = (
                    self.algo_args["train"][
                        "n_rollout_threads"
                    ]
                )
                cur_step = (
                    self.algo_args["train"][
                        "warmup_steps"
                    ]
                    + step * n_threads
                )
                num_env_steps = (
                    self.algo_args["train"][
                        "num_env_steps"
                    ]
                )
                if self.algo_args["eval"]["use_eval"]:
                    print(
                        f"Env {self.args['env']} "
                        f"Task {self.task_name} "
                        f"Algo {self.args['algo']} "
                        f"Exp {self.args['exp_name']}"
                        f" Evaluation at step "
                        f"{cur_step} / "
                        f"{num_env_steps}:"
                    )
                    self.eval(cur_step)
                else:
                    mean_rew = (
                        self.buffer.get_mean_rewards()
                    )
                    print(
                        f"Env {self.args['env']} "
                        f"Task {self.task_name} "
                        f"Algo {self.args['algo']} "
                        f"Exp {self.args['exp_name']}"
                        f" Step {cur_step} / "
                        f"{num_env_steps}, "
                        f"average step reward in "
                        f"buffer: {mean_rew}.\n"
                    )
                    if len(
                        self.done_episodes_rewards
                    ) > 0:
                        aver_episode_rewards = np.mean(
                            self.done_episodes_rewards
                        )
                        print(
                            "Some episodes done, "
                            "average episode reward "
                            "is {}.\n".format(
                                aver_episode_rewards
                            )
                        )
                        self.log_file.write(
                            ",".join(
                                map(
                                    str,
                                    [
                                        cur_step,
                                        aver_episode_rewards,
                                    ],
                                )
                            )
                            + "\n"
                        )
                        self.log_file.flush()
                        self.done_episodes_rewards = []
                self.save()

    def warmup(self):
        """使用随机动作预热经验回放缓冲区。"""
        warmup_steps = (
            self.algo_args["train"]["warmup_steps"]
            // self.algo_args["train"][
                "n_rollout_threads"
            ]
        )
        # obs: (n_threads, n_agents, dim)
        # share_obs: (n_threads, n_agents, dim)
        # available_actions: (threads, n_agents, dim)
        obs, share_obs, available_actions = (
            self.envs.reset()
        )
        for _ in range(warmup_steps):
            # action: (n_threads, n_agents, dim)
            actions = self.sample_actions(
                available_actions
            )
            (
                new_obs,
                new_share_obs,
                rewards,
                dones,
                infos,
                new_available_actions,
            ) = self.envs.step(actions)
            next_obs = new_obs.copy()
            next_share_obs = new_share_obs.copy()
            next_available_actions = (
                new_available_actions.copy()
            )
            aa_shape = np.array(
                available_actions
            ).shape
            data = (
                share_obs,
                obs.transpose(1, 0, 2),
                actions.transpose(1, 0, 2),
                available_actions.transpose(1, 0, 2)
                if len(aa_shape) == 3
                else None,
                rewards,
                dones,
                infos,
                next_share_obs,
                next_obs,
                next_available_actions.transpose(
                    1, 0, 2
                )
                if len(aa_shape) == 3
                else None,
            )
            self.insert(data)
            obs = new_obs
            share_obs = new_share_obs
            available_actions = new_available_actions
        return obs, share_obs, available_actions

    def insert(
        self,
        data,
    ):
        """将数据插入经验回放缓冲区。"""
        (
            # (n_threads, n_agents, share_obs_dim)
            share_obs,
            # (n_agents, n_threads, obs_dim)
            obs,
            # (n_agents, n_threads, action_dim)
            actions,
            # None 或 (n_agents, n_threads, action_number)
            available_actions,
            # (n_threads, n_agents, 1)
            rewards,
            # (n_threads, n_agents)
            dones,
            # 类型: list, 形状: (n_threads, n_agents)
            infos,
            # (n_threads, n_agents, next_share_obs_dim)
            next_share_obs,
            # (n_threads, n_agents, next_obs_dim)
            next_obs,
            # None 或
            # (n_agents, n_threads, next_action_number)
            next_available_actions,
        ) = data

        # 如果所有智能体都结束，则环境结束
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env

        # valid_transition 表示每个转移是否有效
        # （如果对应智能体已死亡则无效）
        # 形状: (n_threads, n_agents, 1)
        valid_transitions = 1 - self.agent_deaths

        self.agent_deaths = np.expand_dims(
            dones, axis=-1
        )

        n_threads = (
            self.algo_args["train"]["n_rollout_threads"]
        )

        # terms 用 False 表示截断，True 表示终止
        if self.state_type == "EP":
            terms = np.full(
                (n_threads, 1), False
            )
            for i in range(n_threads):
                if dones_env[i]:
                    if not (
                        "bad_transition"
                        in infos[i][0].keys()
                        and infos[i][0][
                            "bad_transition"
                        ]
                        is True
                    ):
                        terms[i][0] = True
        elif self.state_type == "FP":
            terms = np.full(
                (n_threads, self.num_agents, 1),
                False,
            )
            for i in range(n_threads):
                for agent_id in range(
                    self.num_agents
                ):
                    if dones[i][agent_id]:
                        if not (
                            "bad_transition"
                            in infos[i][
                                agent_id
                            ].keys()
                            and infos[i][agent_id][
                                "bad_transition"
                            ]
                            is True
                        ):
                            terms[i][agent_id][
                                0
                            ] = True

        for i in range(n_threads):
            if dones_env[i]:
                self.done_episodes_rewards.append(
                    self.train_episode_rewards[i]
                )
                self.train_episode_rewards[i] = 0
                self.agent_deaths = np.zeros(
                    (
                        n_threads,
                        self.num_agents,
                        1,
                    )
                )
                if "original_obs" in infos[i][0]:
                    next_obs[i] = (
                        infos[i][0][
                            "original_obs"
                        ].copy()
                    )
                if "original_state" in infos[i][0]:
                    next_share_obs[i] = (
                        infos[i][0][
                            "original_state"
                        ].copy()
                    )

        if self.state_type == "EP":
            data = (
                # (n_threads, share_obs_dim)
                share_obs[:, 0],
                # (n_agents, n_threads, obs_dim)
                obs,
                # (n_agents, n_threads, action_dim)
                actions,
                # None 或
                # (n_agents, n_threads, action_number)
                available_actions,
                # (n_threads, 1)
                rewards[:, 0],
                # (n_threads, 1)
                np.expand_dims(
                    dones_env, axis=-1
                ),
                # (n_agents, n_threads, 1)
                valid_transitions.transpose(
                    1, 0, 2
                ),
                # (n_threads, 1)
                terms,
                # (n_threads, next_share_obs_dim)
                next_share_obs[:, 0],
                # (n_agents, n_threads, next_obs_dim)
                next_obs.transpose(1, 0, 2),
                # None 或
                # (n_agents, n_threads,
                #  next_action_number)
                next_available_actions,
            )
        elif self.state_type == "FP":
            data = (
                # (n_threads, n_agents,
                #  share_obs_dim)
                share_obs,
                # (n_agents, n_threads, obs_dim)
                obs,
                # (n_agents, n_threads, action_dim)
                actions,
                # None 或
                # (n_agents, n_threads, action_number)
                available_actions,
                # (n_threads, n_agents, 1)
                rewards,
                # (n_threads, n_agents, 1)
                np.expand_dims(dones, axis=-1),
                # (n_agents, n_threads, 1)
                valid_transitions.transpose(
                    1, 0, 2
                ),
                # (n_threads, n_agents, 1)
                terms,
                # (n_threads, n_agents,
                #  next_share_obs_dim)
                next_share_obs,
                # (n_agents, n_threads, next_obs_dim)
                next_obs.transpose(1, 0, 2),
                # None 或
                # (n_agents, n_threads,
                #  next_action_number)
                next_available_actions,
            )

        self.buffer.insert(data)

    def sample_actions(
        self,
        available_actions=None,
    ):
        """采样随机动作用于预热。

        参数:
            available_actions: (np.ndarray) 表示智能体
                可用的动作（如果为 None 则所有动作可用），
                形状为 (n_threads, n_agents,
                action_number) 或 (n_threads, ) of None
        返回:
            actions: (np.ndarray) 采样的动作，
                形状为 (n_threads, n_agents, dim)
        """
        actions = []
        for agent_id in range(self.num_agents):
            action = []
            n_threads = (
                self.algo_args["train"][
                    "n_rollout_threads"
                ]
            )
            for thread in range(n_threads):
                if available_actions[thread] is None:
                    action.append(
                        self.action_spaces[
                            agent_id
                        ].sample()
                    )
                else:
                    action.append(
                        Categorical(
                            torch.tensor(
                                available_actions[
                                    thread,
                                    agent_id,
                                    :,
                                ]
                            )
                        ).sample()
                    )
            actions.append(action)
        space_name = (
            self.envs.action_space[
                agent_id
            ].__class__.__name__
        )
        if space_name == "Discrete":
            return np.expand_dims(
                np.array(actions).transpose(1, 0),
                axis=-1,
            )

        return np.array(actions).transpose(1, 0, 2)

    @torch.no_grad()
    def get_actions(
        self,
        obs,
        available_actions=None,
        add_random=True,
    ):
        """获取用于推演的动作。

        参数:
            obs: (np.ndarray) 输入观测，
                形状为 (n_threads, n_agents, dim)
            available_actions: (np.ndarray) 表示智能体
                可用的动作（如果为 None 则所有动作可用），
                形状为 (n_threads, n_agents,
                action_number) 或 (n_threads, ) of None
            add_random: (bool) 是否添加随机性
        返回:
            actions: (np.ndarray) 智能体动作，
                形状为 (n_threads, n_agents, dim)
        """
        if self.args["algo"] == "hasac":
            actions = []
            for agent_id in range(self.num_agents):
                aa_shape = np.array(
                    available_actions
                ).shape
                if len(aa_shape) == 3:
                    # (n_threads, n_agents,
                    #  action_number)
                    actions.append(
                        _t2n(
                            self.actor[
                                agent_id
                            ].get_actions(
                                obs[:, agent_id],
                                available_actions[
                                    :, agent_id
                                ],
                                add_random,
                            )
                        )
                    )
                else:
                    # (n_threads, ) of None
                    actions.append(
                        _t2n(
                            self.actor[
                                agent_id
                            ].get_actions(
                                obs[:, agent_id],
                                stochastic=add_random,
                            )
                        )
                    )
        else:
            actions = []
            for agent_id in range(self.num_agents):
                actions.append(
                    _t2n(
                        self.actor[
                            agent_id
                        ].get_actions(
                            obs[:, agent_id],
                            add_random,
                        )
                    )
                )
        return np.array(actions).transpose(1, 0, 2)

    def train(self):
        """训练模型。"""
        raise NotImplementedError

    @torch.no_grad()
    def eval(
        self,
        step,
    ):
        """评估模型。"""
        eval_episode_rewards = []
        one_episode_rewards = []
        n_eval_threads = (
            self.algo_args["eval"][
                "n_eval_rollout_threads"
            ]
        )
        for eval_i in range(n_eval_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])
        eval_episode = 0
        if "smac" in self.args["env"]:
            eval_battles_won = 0
        if "football" in self.args["env"]:
            eval_score_cnt = 0
        episode_lens = []
        one_episode_len = np.zeros(
            n_eval_threads, dtype=np.int
        )

        (
            eval_obs,
            eval_share_obs,
            eval_available_actions,
        ) = self.eval_envs.reset()

        while True:
            eval_actions = self.get_actions(
                eval_obs,
                available_actions=(
                    eval_available_actions
                ),
                add_random=False,
            )
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(n_eval_threads):
                one_episode_rewards[eval_i].append(
                    eval_rewards[eval_i]
                )

            one_episode_len += 1

            eval_dones_env = np.all(
                eval_dones, axis=1
            )

            for eval_i in range(n_eval_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    if "smac" in self.args["env"]:
                        if "v2" in self.args["env"]:
                            if eval_infos[eval_i][
                                0
                            ]["battle_won"]:
                                eval_battles_won += 1
                        else:
                            if eval_infos[eval_i][
                                0
                            ]["won"]:
                                eval_battles_won += 1
                    if (
                        "football"
                        in self.args["env"]
                    ):
                        if (
                            eval_infos[eval_i][0][
                                "score_reward"
                            ]
                            > 0
                        ):
                            eval_score_cnt += 1
                    eval_episode_rewards[
                        eval_i
                    ].append(
                        np.sum(
                            one_episode_rewards[
                                eval_i
                            ],
                            axis=0,
                        )
                    )
                    one_episode_rewards[eval_i] = []
                    episode_lens.append(
                        one_episode_len[eval_i].copy()
                    )
                    one_episode_len[eval_i] = 0

            eval_episodes = (
                self.algo_args["eval"][
                    "eval_episodes"
                ]
            )
            if eval_episode >= eval_episodes:
                # eval_log 返回是否应保存当前模型
                eval_episode_rewards = (
                    np.concatenate(
                        [
                            rewards
                            for rewards
                            in eval_episode_rewards
                            if rewards
                        ]
                    )
                )
                eval_avg_rew = np.mean(
                    eval_episode_rewards
                )
                eval_avg_len = np.mean(episode_lens)
                if "smac" in self.args["env"]:
                    win_rate = (
                        eval_battles_won
                        / eval_episode
                    )
                    print(
                        "Eval win rate is {}, "
                        "eval average episode "
                        "rewards is {}, "
                        "eval average episode "
                        "length is {}.".format(
                            win_rate,
                            eval_avg_rew,
                            eval_avg_len,
                        )
                    )
                elif "football" in self.args["env"]:
                    score_rate = (
                        eval_score_cnt
                        / eval_episode
                    )
                    print(
                        "Eval score rate is {}, "
                        "eval average episode "
                        "rewards is {}, "
                        "eval average episode "
                        "length is {}.".format(
                            score_rate,
                            eval_avg_rew,
                            eval_avg_len,
                        )
                    )
                else:
                    print(
                        f"Eval average episode "
                        f"reward is "
                        f"{eval_avg_rew}, "
                        f"eval average episode "
                        f"length is "
                        f"{eval_avg_len}.\n"
                    )
                if "smac" in self.args["env"]:
                    win_rate = (
                        eval_battles_won
                        / eval_episode
                    )
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    win_rate,
                                ],
                            )
                        )
                        + "\n"
                    )
                elif "football" in self.args["env"]:
                    score_rate = (
                        eval_score_cnt
                        / eval_episode
                    )
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                    score_rate,
                                ],
                            )
                        )
                        + "\n"
                    )
                else:
                    self.log_file.write(
                        ",".join(
                            map(
                                str,
                                [
                                    step,
                                    eval_avg_rew,
                                    eval_avg_len,
                                ],
                            )
                        )
                        + "\n"
                    )
                self.log_file.flush()
                self.writter.add_scalar(
                    "eval_average_episode_rewards",
                    eval_avg_rew,
                    step,
                )
                self.writter.add_scalar(
                    "eval_average_episode_length",
                    eval_avg_len,
                    step,
                )
                break

    @torch.no_grad()
    def render(self):
        """渲染模型。"""
        print("start rendering")
        if self.manual_expand_dims:
            # 此环境需要手动扩展并行环境数维度
            render_episodes = (
                self.algo_args["render"][
                    "render_episodes"
                ]
            )
            for _ in range(render_episodes):
                (
                    eval_obs,
                    _,
                    eval_available_actions,
                ) = self.envs.reset()
                eval_obs = np.expand_dims(
                    np.array(eval_obs), axis=0
                )
                eval_available_actions = np.array(
                    [eval_available_actions]
                )
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=(
                            eval_available_actions
                        ),
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(
                        eval_actions[0]
                    )
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(
                        np.array(eval_obs), axis=0
                    )
                    eval_available_actions = (
                        np.array(
                            [eval_available_actions]
                        )
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(
                            "total reward of this "
                            "episode: "
                            f"{rewards}"
                        )
                        break
        else:
            # 此环境不需要手动扩展并行环境数维度，
            # 例如 dexhands，它实例化了64对手的
            # 并行环境
            render_episodes = (
                self.algo_args["render"][
                    "render_episodes"
                ]
            )
            for _ in range(render_episodes):
                (
                    eval_obs,
                    _,
                    eval_available_actions,
                ) = self.envs.reset()
                rewards = 0
                while True:
                    eval_actions = self.get_actions(
                        eval_obs,
                        available_actions=(
                            eval_available_actions
                        ),
                        add_random=False,
                    )
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += (
                        eval_rewards[0][0][0]
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(
                            "total reward of this "
                            "episode: "
                            f"{rewards}"
                        )
                        break
        if "smac" in self.args["env"]:
            # smac 的回放，不进行渲染
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """恢复模型。"""
        model_dir = (
            self.algo_args["train"]["model_dir"]
        )
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(
                model_dir, agent_id
            )
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(model_dir)
            if self.value_normalizer is not None:
                vn_path = (
                    str(model_dir)
                    + "/value_normalizer"
                    + ".pt"
                )
                vn_state_dict = torch.load(
                    vn_path,
                    map_location=self.device,
                )
                self.value_normalizer.load_state_dict(
                    vn_state_dict
                )

    def save(self):
        """保存模型。"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(
                self.save_dir, agent_id
            )
        self.critic.save(self.save_dir)
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir)
                + "/value_normalizer"
                + ".pt",
            )

    def close(self):
        """关闭环境、写入器和日志文件。"""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if (
                self.algo_args["eval"]["use_eval"]
                and self.eval_envs is not self.envs
            ):
                self.eval_envs.close()
            self.writter.export_scalars_to_json(
                str(self.log_dir + "/summary.json")
            )
            self.writter.close()
            self.log_file.close()
