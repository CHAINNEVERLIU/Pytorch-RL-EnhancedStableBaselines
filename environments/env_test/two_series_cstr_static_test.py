import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from environments.cstr.twoseriescstr import TwoSeriesCSTREnv


def steady_state_analysis(env):
    """
    稳态分析：检查系统在不同初始条件下是否能达到稳定状态
    """
    # 尝试不同的初始状态和冷却水流量
    initial_states = [
        np.array([0.45, 310.0, 0.25, 290.0]),  # 默认初始状态
        np.array([0.1, 250, 0.005, 280]),  # 低浓度低温
        np.array([0.9, 400, 0.009, 420])  # 高浓度高温
    ]

    actions = [
        np.array([50, 50]),  # 低流量
        np.array([100, 100]),  # 中等流量
        np.array([200, 200])  # 高流量
    ]

    results = []

    for init_state in initial_states:
        for action in actions:
            env.reset()
            env.state = env._normalize_state(init_state)

            # 模拟多个步骤
            state_history = [init_state]
            for _ in range(1000):
                normalized_action = env._normalize_action(action)
                next_state, _, terminated, truncated, _ = env.step(normalized_action)
                current_state = env._denormalize_state(next_state)
                state_history.append(current_state)

                if terminated or truncated:
                    break

            # 分析最后几个状态的变化
            final_states = state_history[-5:]
            state_variation = np.std(final_states, axis=0)
            results.append({
                # 'initial_state': init_state,
                # 'action': action,
                'final state': state_history[-1:],
                # 'state_variation': state_variation
            })

    return results


def dynamics_behavior_analysis(env):
    """
    分析系统动力学行为
    - 检查不同动作对系统状态的影响
    - 验证反应速率、温度变化的合理性
    """

    def simulate_trajectory(initial_state, actions):
        env.reset()
        env.state = env._normalize_state(initial_state)

        trajectories = {
            'concentration': [initial_state[0], initial_state[2]],
            'temperature': [initial_state[1], initial_state[3]]
        }

        for action in actions:
            normalized_action = env._normalize_action(action)
            next_state, _, _, _, _ = env.step(normalized_action)
            current_state = env._denormalize_state(next_state)

            trajectories['concentration'].append(current_state[0])
            trajectories['concentration'].append(current_state[2])
            trajectories['temperature'].append(current_state[1])
            trajectories['temperature'].append(current_state[3])

        return trajectories

    # 不同场景
    scenarios = [
        {
            'initial_state': np.array([0.5, 300, 0.001, 300]),
            'actions': [
                np.array([50, 50]),
                np.array([100, 100]),
                np.array([200, 200])
            ]
        },
        # 可以添加更多场景
    ]

    results = []
    for scenario in scenarios:
        trajectory = simulate_trajectory(
            scenario['initial_state'],
            scenario['actions']
        )
        results.append(trajectory)

    return results


def boundary_conditions_test(env):
    """
    测试极端边界条件
    """
    # 极端初始状态
    extreme_states = [
        np.array([0, 100, 0, 100]),  # 最低值
        np.array([1, 600, 1, 600]),  # 最高值
    ]

    # 极端动作
    extreme_actions = [
        np.array([10, 10]),  # 最小流量
        np.array([200, 200])  # 最大流量
    ]

    results = []

    for state in extreme_states:
        for action in extreme_actions:
            env.reset()
            env.state = env._normalize_state(state)

            try:
                next_state, reward, terminated, truncated, info = env.step(env._normalize_action(action))
                results.append({
                    'initial_state': state,
                    'action': action,
                    'next_state': env._denormalize_state(next_state),
                    'reward': reward,
                    'status': 'Normal'
                })
            except Exception as e:
                results.append({
                    'initial_state': state,
                    'action': action,
                    'error': str(e),
                    'status': 'Error'
                })

    return results


# 创建环境并模拟
env = TwoSeriesCSTREnv()
stability_results = steady_state_analysis(env)
# 打印最终稳态
print("Final Steady State (Normalized):")
print(stability_results)

# results = dynamics_behavior_analysis(env)
# # 打印最终稳态
# print("Final Dynamics Behavior Analysis Results:")
# print(results)


# results = boundary_conditions_test(env)
# # 打印最终稳态
# print("Final Boundary Conditions Test:")
# print(results)









