"""Baselines 训练入口脚本。

用法:
    python -m scripts.train_baselines --algo mappo --env mamujoco
    python -m scripts.train_baselines --algo mappo --env mamujoco --exp_name debug
    python -m scripts.train_baselines --algo hasac --env mamujoco --task run_bwd

所有 baselines/configs/algos/*.yaml 和 baselines/configs/envs/*.yaml 中的
参数均可通过 --key value 的形式在命令行覆盖。
"""

import argparse

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
        default="installtest",
        help="实验名称，用于区分同一算法/环境下的不同实验",
    )

    args, unparsed_args = parser.parse_known_args()

    # 加载 yaml 默认配置
    algo_args, env_args = get_defaults_yaml_args(
        args.algo, args.env
    )

    # 解析命令行中的额外 --key value 参数
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = dict(zip(keys, values))

    # 用命令行参数覆盖 yaml 配置
    update_args(unparsed_dict, algo_args, env_args)

    # 构建 runner 所需的 args dict
    args_dict = {
        "algo": args.algo,
        "env": args.env,
        "exp_name": args.exp_name,
    }

    # 启动训练
    runner = RUNNER_REGISTRY[args.algo](
        args_dict, algo_args, env_args
    )

    if algo_args["render"]["use_render"]:
        runner.render()
    else:
        runner.run()

    runner.close()


def process(arg):
    """尝试将字符串参数转换为 Python 字面量。

    使用 ast.literal_eval 替代 eval，仅解析数值、布尔等
    安全字面量，避免 eval 将 "9|8" 等含运算符的字符串
    误解析为表达式。
    """
    import ast
    try:
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError):
        return arg


if __name__ == "__main__":
    main()
