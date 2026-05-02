"""配置文件的加载与更新工具。"""

import json
import os
import time
from json import JSONDecodeError

import yaml


def _deep_update(
    base,
    overrides,
):
    """递归地用 overrides 中的值更新 base 字典。

    参数:
        base: 待更新的基础字典。
        overrides: 覆盖值字典。
    """
    for k, v in overrides.items():
        if (
            isinstance(v, dict)
            and isinstance(base.get(k), dict)
        ):
            _deep_update(base[k], v)
        else:
            base[k] = v


def get_defaults_yaml_args(
    algo,
    env,
    scenario,
):
    """根据算法、环境和场景加载配置文件。

    环境配置文件位于 configs/envs/{env}/{scenario}.yaml，
    例如 configs/envs/mamujoco/cheetah.yaml。

    参数:
        algo: 算法名称。
        env: 环境名称。
        scenario: 场景名称，用于选择环境子配置文件。
    返回:
        algo_args: 算法配置字典。
        env_args: 环境配置字典。
    """
    base_path = os.path.split(
        os.path.dirname(os.path.abspath(__file__))
    )[0]
    algo_cfg_path = os.path.join(
        base_path, "configs", "algos", f"{algo}.yaml",
    )
    env_cfg_path = os.path.join(
        base_path, "configs", "envs",
        env, f"{scenario.lower()}.yaml",
    )

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)

    # 应用环境配置中的算法参数覆盖
    overrides = env_args.pop("algo_overrides", None)
    if overrides:
        _deep_update(algo_args, overrides)

    return algo_args, env_args


def update_args(
    unparsed_dict,
    *args,
):
    """使用未解析的命令行参数更新已加载的配置。

    参数:
        unparsed_dict: 未解析的命令行参数字典。
        *args: 待更新的参数字典列表。
    """

    def update_dict(
        dict1,
        dict2,
    ):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(
    env,
    env_args,
):
    """根据环境名称获取任务名称。"""
    if env == "mamujoco":
        task = "mamujoco"
    return task


def init_dir(
    env,
    env_args,
    algo,
    seed,
    logger_path,
):
    """初始化结果保存目录。"""
    task = get_task_name(env, env_args)
    hms_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime()
    )
    results_path = os.path.join(
        logger_path,
        env,
        task,
        algo,
        "-".join(
            ["seed-{:0>5}".format(seed), hms_time]
        ),
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writter = SummaryWriter(log_path)
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter


def is_json_serializable(value):
    """检查值是否可以被JSON序列化。"""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, JSONDecodeError):
        return False


def convert_json(obj):
    """将对象转换为可被JSON序列化的版本。"""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {
                convert_json(k): convert_json(v)
                for k, v in obj.items()
            }

        elif isinstance(obj, tuple):
            return tuple(
                convert_json(x) for x in obj
            )

        elif isinstance(obj, list):
            return [
                convert_json(x) for x in obj
            ]

        elif hasattr(obj, "__name__") and not (
            "lambda" in obj.__name__
        ):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v)
                for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(
    args,
    algo_args,
    env_args,
    run_dir,
):
    """保存程序的配置信息。"""
    config = {
        "main_args": args,
        "algo_args": algo_args,
        "env_args": env_args,
    }
    config_json = convert_json(config)
    output = json.dumps(
        config_json,
        separators=(",", ":\t"),
        indent=4,
        sort_keys=True,
    )
    with open(
        os.path.join(run_dir, "config.json"),
        "w",
        encoding="utf-8",
    ) as out:
        out.write(output)
