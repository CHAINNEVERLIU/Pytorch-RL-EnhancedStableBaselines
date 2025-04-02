import gymnasium as gym
import numpy as np
import os
import torch as th
from stable_baselines3 import TD3
from core.common.buffers import ReplayBuffer
from core.common.vec_env import VecNormalize, DummyVecEnv
from tqdm import tqdm
import pickle
import h5py
import json
from typing import Dict, Any
from core.common.env_util import make_vec_env
from core.common.save_util import load_from_pkl, save_to_pkl


def collect_offline_dataset(
        model_path,
        env_id,
        num_episodes=100,
        random_action_prob=0.1,
        dataset_path="./offline_data",
        dataset_name="td3_halfcheetah_dataset",
        use_mixed_policy=True
):
    """
    使用训练好的TD3模型通过SB3的ReplayBuffer收集离线数据集

    参数:
        model_path: 已训练好的TD3模型路径
        vec_normalize_path: VecNormalize保存的路径（可选）
        env_id: 环境ID
        num_episodes: 要收集的回合数
        random_action_prob: 选择随机动作的概率（探索）
        dataset_path: 数据集保存路径
        dataset_name: 数据集名称
        use_mixed_policy: 是否混合使用训练好的策略和随机策略
    """
    # 创建保存数据的目录
    os.makedirs(dataset_path, exist_ok=True)

    # 加载模型
    model = TD3.load(model_path)

    # 创建环境
    vec_env = DummyVecEnv([lambda: gym.make(env_id)])  # 重新创建评估环境
    vec_env.training = False  # 不更新统计数据
    vec_env.norm_reward = False  # 不归一化奖励

    # 使用和模型训练时相同的设备
    device = model.device

    # 创建ReplayBuffer
    buffer_size = num_episodes * 1000  # 估计所需的缓冲区大小
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        device=device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True
    )

    episode_rewards = []
    total_transitions = 0

    # 收集数据
    with tqdm(total=num_episodes, desc="收集离线数据") as pbar:
        episode_count = 0
        while episode_count < num_episodes:
            obs = vec_env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                # 决定是使用策略还是随机动作
                if use_mixed_policy and np.random.random() < random_action_prob:
                    # 随机动作（探索）
                    action = vec_env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                # 执行动作
                next_obs, reward, done, info = vec_env.step(action)

                # 添加到ReplayBuffer
                replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos=info  # ReplayBuffer期望一个列表
                )

                obs = next_obs
                episode_reward += reward[0]
                episode_steps += 1
                total_transitions += 1

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_count += 1
            pbar.update(1)
            pbar.set_postfix({"reward": f"{episode_reward:.2f}", "steps": episode_steps})

            # 保存ReplayBuffer为pickle格式
        buffer_path = f"{dataset_path}/{dataset_name}_buffer.pkl"
        save_to_pkl(buffer_path, replay_buffer)

        # 生成统计信息
        stats = {
                'num_episodes': num_episodes,
                'total_transitions': total_transitions,
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'min_reward': float(np.min(episode_rewards)),
                'max_reward': float(np.max(episode_rewards)),
            }

        # 保存统计信息为JSON
        stats_path = f"{dataset_path}/{dataset_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"\n离线数据集收集完成!")
        print(f"总转移数: {total_transitions}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"奖励范围: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")

        print(f"\n数据已保存为以下格式:")
        print(f"1. SB3 ReplayBuffer (pickle): {buffer_path}")
        print(f"3. 统计信息 (JSON): {stats_path}")

        return replay_buffer, stats


if __name__ == "__main__":
    # 配置参数
    model_path = "./models/td3_halfcheetah_2025-03-25_21-24-28/td3_halfcheetah_final.zip"  # 修改为你的模型路径
    env_id = "HalfCheetah-v5"

    # 收集三种不同策略的数据集

    # # 1. 混合策略数据集（训练好的策略 + 一些随机动作）
    # mixed_buffer, mixed_stats = collect_offline_dataset(
    #     model_path=model_path,
    #     vec_normalize_path=vec_normalize_path,
    #     env_id=env_id,
    #     num_episodes=100,
    #     random_action_prob=0.1,
    #     dataset_path="./offline_data",
    #     dataset_name="td3_halfcheetah_mixed",
    #     use_mixed_policy=True
    # )

    # 2. 专家策略数据集（仅使用训练好的策略）
    expert_buffer, expert_stats = collect_offline_dataset(
        model_path=model_path,
        env_id=env_id,
        num_episodes=100,
        random_action_prob=0.0,
        dataset_path="./offline_data",
        dataset_name="td3_halfcheetah_expert",
        use_mixed_policy=False
    )


