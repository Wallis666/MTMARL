"""MaMuJoCo 多任务环境的 Logger。"""

from baselines.common.base_logger import BaseLogger


class MAMuJoCoLogger(BaseLogger):
    """
    将任务名称编码进 task_name，便于 TensorBoard 曲线与保存目录区分
    不同多任务设置下的实验结果。
    """

    def get_task_name(self) -> str:
        """
        组合 agent_conf 与任务名称作为实验标识。

        返回:
            形如 "HalfCheetah-2x3-run_fwd" 的字符串。
        """
        scenario = self.env_args.get("scenario", "HalfCheetah")
        agent_conf = self.env_args.get("agent_conf", "unknown")
        task = self.env_args.get("task", "run_fwd")
        return f"{scenario}-{agent_conf}-{task}"
