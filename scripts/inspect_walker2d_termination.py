"""
Walker2d 终止条件检查脚本。

检查 gymnasium-robotics Walker2d 环境自带的终止条件，
包括 healthy 范围、terminate_when_unhealthy 等参数。
"""

import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import (
    MultiAgentMujocoEnv,
)


def main() -> None:
    """检查 Walker2d 终止条件。"""
    env = MultiAgentMujocoEnv(
        scenario="Walker2d",
        agent_conf="2x3",
        agent_obsk=0,
        render_mode=None,
    )

    # 获取底层单智能体环境
    single_env = env.single_agent_env
    unwrapped = single_env.unwrapped

    print("=" * 60)
    print("Walker2d 终止条件检查")
    print("=" * 60)

    # 检查环境类名和继承链
    print(f"\n底层环境类: {type(unwrapped).__name__}")
    print(f"继承链: {[c.__name__ for c in type(unwrapped).__mro__]}")

    # 检查所有可能与终止相关的属性
    print("\n--- 终止相关属性 ---")
    attrs_to_check = [
        "terminate_when_unhealthy",
        "_terminate_when_unhealthy",
        "healthy_z_range",
        "_healthy_z_range",
        "healthy_angle_range",
        "_healthy_angle_range",
        "_healthy_reward",
        "healthy_reward",
        "_healthy_state_range",
        "healthy_state_range",
    ]
    for attr in attrs_to_check:
        if hasattr(unwrapped, attr):
            val = getattr(unwrapped, attr)
            print(f"  {attr} = {val}")
        else:
            print(f"  {attr} = (不存在)")

    # 检查 is_healthy 属性/方法
    print("\n--- 健康状态检查 ---")
    if hasattr(unwrapped, "is_healthy"):
        print(f"  is_healthy = {unwrapped.is_healthy}")
    if hasattr(unwrapped, "healthy"):
        print(f"  healthy = {unwrapped.healthy}")
    if hasattr(unwrapped, "_is_healthy"):
        print(f"  _is_healthy = {unwrapped._is_healthy}")

    # 查看 terminated 方法源码
    print("\n--- step 方法检查 ---")
    import inspect
    # 获取 step 方法的源码
    try:
        step_src = inspect.getsource(type(unwrapped).step)
        print("底层 step 方法源码:")
        print(step_src[:2000])
    except (TypeError, OSError) as e:
        print(f"  无法获取 step 源码: {e}")

    # 检查 terminated 方法
    print("\n--- terminated 方法 ---")
    for method_name in ["terminated", "_get_terminated",
                         "is_terminated", "_is_terminated"]:
        if hasattr(unwrapped, method_name):
            try:
                src = inspect.getsource(
                    getattr(type(unwrapped), method_name)
                )
                print(f"\n{method_name} 源码:")
                print(src[:1000])
            except (TypeError, OSError):
                val = getattr(unwrapped, method_name)
                print(f"  {method_name} = {val}")

    # 实际运行检测终止行为
    print("\n--- 实际终止行为测试 ---")
    obs, _ = env.reset()
    terminated_count = 0
    for step_i in range(500):
        actions = {
            agent: env.action_space(agent).sample()
            for agent in obs
        }
        obs, rewards, terms, truncs, infos = env.step(actions)

        if any(terms.values()):
            terminated_count += 1
            # 打印终止时的状态
            qpos = unwrapped.data.qpos
            height = qpos[1]
            angle = qpos[2]
            angle_deg = np.rad2deg(angle)
            info = next(iter(infos.values()))
            print(
                f"  第 {step_i + 1} 步终止: "
                f"height={height:.3f}  "
                f"angle={angle_deg:+.1f}°  "
                f"info={info}"
            )
            if terminated_count >= 5:
                break
            obs, _ = env.reset()

    print(f"\n共终止 {terminated_count} 次（500 步内）")

    # 检查 MultiAgentMujocoEnv 的 step 方法
    print("\n--- MultiAgentMujocoEnv step 源码 ---")
    try:
        ma_step = inspect.getsource(MultiAgentMujocoEnv.step)
        print(ma_step[:2000])
    except (TypeError, OSError) as e:
        print(f"  无法获取源码: {e}")

    env.close()


if __name__ == "__main__":
    main()
