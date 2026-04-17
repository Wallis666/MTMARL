"""同策略算法的基础运行器模块。"""

import time

import numpy as np
import setproctitle
import torch

from baselines.algos.actors import ALGO_REGISTRY
from baselines.algos.critics.v_critic import VCritic
from baselines.common.buffers.on_policy_actor_buffer import (
    OnPolicyActorBuffer,
)
from baselines.common.buffers.on_policy_critic_buffer_ep import (
    OnPolicyCriticBufferEP,
)
from baselines.common.buffers.on_policy_critic_buffer_fp import (
    OnPolicyCriticBufferFP,
)
from baselines.common.value_norm import ValueNorm
from baselines.envs import LOGGER_REGISTRY
from baselines.utils.config import init_dir, save_config
from baselines.utils.env import (
    get_num_agents,
    make_eval_env,
    make_render_env,
    make_train_env,
    set_seed,
)
from baselines.utils.model import init_device
from baselines.utils.trans import _t2n


class OnPolicyBaseRunner:
    """同策略算法的基础运行器。"""

    def __init__(
        self,
        args,
        algo_args,
        env_args,
    ):
        """初始化 OnPolicyBaseRunner 类。

        参数:
            args: 由 argparse 解析的命令行参数。
                包含三个键: algo, env, exp_name。
            algo_args: 与算法相关的参数，从配置文件加载
                并使用未解析的命令行参数更新。
            env_args: 与环境相关的参数，从配置文件加载
                并使用未解析的命令行参数更新。
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = (
            algo_args["algo"]["action_aggregation"]
        )
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:
            # 训练模式，非渲染模式
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
        # 设置进程标题
        setproctitle.setproctitle(
            str(env_args.get("scenario", args["env"]))
            + "-"
            + str(args["algo"])
            + "-"
            + str(env_args.get("task", ""))
        )

        # 设置环境配置
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

        # 演员（actor）
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
                    "参数共享无效。"
                )
                assert (
                    self.envs.action_space[agent_id]
                    == self.envs.action_space[0]
                ), (
                    "智能体具有异构动作空间，"
                    "参数共享无效。"
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

        if self.algo_args["render"]["use_render"] is False:
            # 训练模式，非渲染模式
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {
                        **algo_args["train"],
                        **algo_args["model"],
                    },
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            share_observation_space = (
                self.envs.share_observation_space[0]
            )
            self.critic = VCritic(
                {
                    **algo_args["model"],
                    **algo_args["algo"],
                },
                share_observation_space,
                device=self.device,
            )
            if self.state_type == "EP":
                # EP 表示环境提供（Environment Provided），
                # 如 MAPPO 论文所述。
                # 在 EP 模式下，所有智能体的全局状态相同。
                self.critic_buffer = (
                    OnPolicyCriticBufferEP(
                        {
                            **algo_args["train"],
                            **algo_args["model"],
                            **algo_args["algo"],
                        },
                        share_observation_space,
                    )
                )
            elif self.state_type == "FP":
                # FP 表示特征裁剪（Feature Pruned），
                # 如 MAPPO 论文所述。
                # 在 FP 模式下，所有智能体的全局状态不同，
                # 因此需要智能体数量的维度。
                self.critic_buffer = (
                    OnPolicyCriticBufferFP(
                        {
                            **algo_args["train"],
                            **algo_args["model"],
                            **algo_args["algo"],
                        },
                        share_observation_space,
                        self.num_agents,
                    )
                )
            else:
                raise NotImplementedError

            if (
                self.algo_args["train"]["use_valuenorm"]
                is True
            ):
                self.value_normalizer = ValueNorm(
                    1, device=self.device
                )
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args["env"]](
                args,
                algo_args,
                env_args,
                self.num_agents,
                self.writter,
                self.run_dir,
            )
        if self.algo_args["train"]["model_dir"] is not None:
            # 恢复模型
            self.restore()

    def run(self):
        """运行训练（或渲染）流程。"""
        if (
            self.algo_args["render"]["use_render"]
            is True
        ):
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"][
                "n_rollout_threads"
            ]
        )

        # 训练开始时的日志回调
        self.logger.init(episodes)

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:
                # 学习率线性衰减
                if self.share_param:
                    self.actor[0].lr_decay(
                        episode, episodes
                    )
                else:
                    for agent_id in range(
                        self.num_agents
                    ):
                        self.actor[agent_id].lr_decay(
                            episode, episodes
                        )
                self.critic.lr_decay(episode, episodes)

            # 每个回合开始时的日志回调
            self.logger.episode_init(episode)

            # 切换到评估模式
            self.prep_rollout()
            for step in range(
                self.algo_args["train"]["episode_length"]
            ):
                # 从演员采样动作，从评论家采样价值
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs:
                #     (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) 为 None
                #     或 (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # 每步的日志回调
                self.logger.per_step(data)

                # 将数据插入缓冲区
                self.insert(data)

            # 计算回报并更新网络
            self.compute()
            # 切换到训练模式
            self.prep_training()

            actor_train_infos, critic_train_info = (
                self.train()
            )

            # 记录信息
            if (
                episode
                % self.algo_args["train"]["log_interval"]
                == 0
            ):
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # 评估
            if (
                episode
                % self.algo_args["train"][
                    "eval_interval"
                ]
                == 0
            ):
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def warmup(self):
        """预热回放缓冲区。"""
        # 重置环境
        obs, share_obs, available_actions = (
            self.envs.reset()
        )
        # 回放缓冲区
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = (
                obs[:, agent_id].copy()
            )
            if (
                self.actor_buffer[agent_id]
                .available_actions
                is not None
            ):
                self.actor_buffer[
                    agent_id
                ].available_actions[0] = (
                    available_actions[
                        :, agent_id
                    ].copy()
                )
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = (
                share_obs[:, 0].copy()
            )
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = (
                share_obs.copy()
            )

    @torch.no_grad()
    def collect(
        self,
        step,
    ):
        """从演员和评论家收集动作和价值。

        参数:
            step: 回合中的步数。
        返回:
            values, actions, action_log_probs,
            rnn_states, rnn_states_critic
        """
        # 从 n 个演员收集 actions, action_log_probs,
        # rnn_states
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = (
                self.actor[agent_id].get_actions(
                    self.actor_buffer[agent_id].obs[
                        step
                    ],
                    self.actor_buffer[
                        agent_id
                    ].rnn_states[step],
                    self.actor_buffer[agent_id].masks[
                        step
                    ],
                    self.actor_buffer[
                        agent_id
                    ].available_actions[step]
                    if self.actor_buffer[agent_id]
                    .available_actions
                    is not None
                    else None,
                )
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(
                _t2n(action_log_prob)
            )
            rnn_state_collector.append(
                _t2n(rnn_state)
            )
        # (n_agents, n_threads, dim)
        #     -> (n_threads, n_agents, dim)
        actions = np.array(
            action_collector
        ).transpose(1, 0, 2)
        action_log_probs = np.array(
            action_log_prob_collector
        ).transpose(1, 0, 2)
        rnn_states = np.array(
            rnn_state_collector
        ).transpose(1, 0, 2, 3)

        # 从 1 个评论家收集 values, rnn_states_critic
        if self.state_type == "EP":
            value, rnn_state_critic = (
                self.critic.get_values(
                    self.critic_buffer.share_obs[step],
                    self.critic_buffer.rnn_states_critic[
                        step
                    ],
                    self.critic_buffer.masks[step],
                )
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = (
                self.critic.get_values(
                    np.concatenate(
                        self.critic_buffer.share_obs[
                            step
                        ]
                    ),
                    np.concatenate(
                        self.critic_buffer
                        .rnn_states_critic[step]
                    ),
                    np.concatenate(
                        self.critic_buffer.masks[step]
                    ),
                )
            )
            # 将 (n_threads * n_agents, dim) 拆分为
            # (n_threads, n_agents, dim)
            n_threads = self.algo_args["train"][
                "n_rollout_threads"
            ]
            values = np.array(
                np.split(_t2n(value), n_threads)
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic),
                    n_threads,
                )
            )

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        )

    def insert(
        self,
        data,
    ):
        """将数据插入缓冲区。"""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # 如果所有智能体都完成，则环境完成
        dones_env = np.all(dones, axis=1)

        # 如果环境完成，将 rnn_state 重置为全零
        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # 如果环境完成，将 rnn_state_critic 重置为全零
        if self.state_type == "EP":
            rnn_states_critic[
                dones_env == True
            ] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[
                dones_env == True
            ] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        # masks 使用 0 来屏蔽刚结束的线程。
        # 用于标记何时应重置 rnn 状态
        n_threads = self.algo_args["train"][
            "n_rollout_threads"
        ]
        masks = np.ones(
            (n_threads, self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        # active_masks 使用 0 来屏蔽已死亡的智能体
        active_masks = np.ones(
            (n_threads, self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1),
            dtype=np.float32,
        )
        active_masks[dones_env == True] = np.ones(
            (
                (dones_env == True).sum(),
                self.num_agents,
                1,
            ),
            dtype=np.float32,
        )

        # bad_masks 使用 0 表示截断，1 表示终止
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] is True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition"
                        in info[agent_id].keys()
                        and info[agent_id][
                            "bad_transition"
                        ]
                        is True
                        else [1.0]
                        for agent_id in range(
                            self.num_agents
                        )
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs,
                rnn_states_critic,
                values,
                rewards,
                masks,
                bad_masks,
            )

    @torch.no_grad()
    def compute(self):
        """计算回报和优势。

        计算评论家对最后状态的评估，
        然后让缓冲区计算回报，
        这些回报将在训练期间使用。
        """
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[
                    -1
                ],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(
                    self.critic_buffer.share_obs[-1]
                ),
                np.concatenate(
                    self.critic_buffer
                    .rnn_states_critic[-1]
                ),
                np.concatenate(
                    self.critic_buffer.masks[-1]
                ),
            )
            n_threads = self.algo_args["train"][
                "n_rollout_threads"
            ]
            next_value = np.array(
                np.split(
                    _t2n(next_value), n_threads
                )
            )
        self.critic_buffer.compute_returns(
            next_value, self.value_normalizer
        )

    def train(self):
        """训练模型。"""
        raise NotImplementedError

    def after_update(self):
        """更新后执行必要的数据操作。

        更新后，将最后一步的数据复制到缓冲区的
        第一个位置。这将用于生成新的动作。
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """评估模型。"""
        # 评估开始时的日志回调
        self.logger.eval_init()
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = (
            self.eval_envs.reset()
        )

        n_eval_threads = self.algo_args["eval"][
            "n_eval_rollout_threads"
        ]
        eval_rnn_states = np.zeros(
            (
                n_eval_threads,
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (n_eval_threads, self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = (
                    self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[
                            :, agent_id
                        ],
                        eval_masks[:, agent_id],
                        eval_available_actions[
                            :, agent_id
                        ]
                        if eval_available_actions[0]
                        is not None
                        else None,
                        deterministic=True,
                    )
                )
                eval_rnn_states[:, agent_id] = (
                    _t2n(temp_rnn_state)
                )
                eval_actions_collector.append(
                    _t2n(eval_actions)
                )

            eval_actions = np.array(
                eval_actions_collector
            ).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            # 评估每步的日志回调
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(
                eval_dones, axis=1
            )

            # 如果环境完成，将 rnn_state 重置为全零
            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (
                    n_eval_threads,
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )
            eval_masks[
                eval_dones_env == True
            ] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )

            for eval_i in range(n_eval_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    # 回合完成时的日志回调
                    self.logger.eval_thread_done(
                        eval_i
                    )

            if (
                eval_episode
                >= self.algo_args["eval"][
                    "eval_episodes"
                ]
            ):
                # 评估结束时的日志回调
                self.logger.eval_log(eval_episode)
                break

    @torch.no_grad()
    def render(self):
        """渲染模型。"""
        print("start rendering")
        if self.manual_expand_dims:
            # 此环境需要手动扩展并行环境数量维度
            for _ in range(
                self.algo_args["render"][
                    "render_episodes"
                ]
            ):
                eval_obs, _, eval_available_actions = (
                    self.envs.reset()
                )
                eval_obs = np.expand_dims(
                    np.array(eval_obs), axis=0
                )
                eval_available_actions = (
                    np.expand_dims(
                        np.array(
                            eval_available_actions
                        ),
                        axis=0,
                    )
                    if eval_available_actions
                    is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (
                        self.env_num,
                        self.num_agents,
                        1,
                    ),
                    dtype=np.float32,
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(
                        self.num_agents
                    ):
                        (
                            eval_actions,
                            temp_rnn_state,
                        ) = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[
                                :, agent_id
                            ],
                            eval_masks[
                                :, agent_id
                            ],
                            eval_available_actions[
                                :, agent_id
                            ]
                            if eval_available_actions
                            is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[
                            :, agent_id
                        ] = _t2n(temp_rnn_state)
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
                    ) = self.envs.step(
                        eval_actions[0]
                    )
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(
                        np.array(eval_obs), axis=0
                    )
                    eval_available_actions = (
                        np.expand_dims(
                            np.array(
                                eval_available_actions
                            ),
                            axis=0,
                        )
                        if eval_available_actions
                        is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(
                            "total reward of this "
                            f"episode: {rewards}"
                        )
                        break
        else:
            # 此环境不需要手动扩展并行环境数量维度，
            # 例如 dexhands，它实例化了一个 64 对手的
            # 并行环境
            for _ in range(
                self.algo_args["render"][
                    "render_episodes"
                ]
            ):
                eval_obs, _, eval_available_actions = (
                    self.envs.reset()
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (
                        self.env_num,
                        self.num_agents,
                        1,
                    ),
                    dtype=np.float32,
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(
                        self.num_agents
                    ):
                        (
                            eval_actions,
                            temp_rnn_state,
                        ) = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[
                                :, agent_id
                            ],
                            eval_masks[
                                :, agent_id
                            ],
                            eval_available_actions[
                                :, agent_id
                            ]
                            if eval_available_actions[
                                0
                            ]
                            is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[
                            :, agent_id
                        ] = _t2n(temp_rnn_state)
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
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(
                            "total reward of this "
                            f"episode: {rewards}"
                        )
                        break
        if "smac" in self.args["env"]:
            # smac 的回放，无渲染
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def prep_rollout(self):
        """准备进行数据采集。"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """准备进行训练。"""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()

    def save(self):
        """保存模型参数。"""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir)
                + "/actor_agent"
                + str(agent_id)
                + ".pt",
            )
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(),
            str(self.save_dir)
            + "/critic_agent"
            + ".pt",
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir)
                + "/value_normalizer"
                + ".pt",
            )

    def restore(self):
        """恢复模型参数。"""
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(
                    self.algo_args["train"][
                        "model_dir"
                    ]
                )
                + "/actor_agent"
                + str(agent_id)
                + ".pt",
                map_location=self.device,
            )
            self.actor[agent_id].actor.load_state_dict(
                policy_actor_state_dict
            )
        if not self.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(
                    self.algo_args["train"][
                        "model_dir"
                    ]
                )
                + "/critic_agent"
                + ".pt",
                map_location=self.device,
            )
            self.critic.critic.load_state_dict(
                policy_critic_state_dict
            )
            if self.value_normalizer is not None:
                value_normalizer_state_dict = (
                    torch.load(
                        str(
                            self.algo_args["train"][
                                "model_dir"
                            ]
                        )
                        + "/value_normalizer"
                        + ".pt",
                        map_location=self.device,
                    )
                )
                self.value_normalizer.load_state_dict(
                    value_normalizer_state_dict
                )

    def close(self):
        """关闭环境、写入器和日志记录器。"""
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
            self.logger.close()
