import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"./logs/td3_halfcheetah_{timestamp}"
model_dir = f"./models/td3_halfcheetah_{timestamp}"
tensorboard_log = f"./tensorboard/td3_halfcheetah_{timestamp}"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)

# 配置日志记录器
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# 创建向量化环境（支持并行训练多个环境实例）
env_id = "HalfCheetah-v5"
vec_env = DummyVecEnv([lambda: gym.make(env_id)])

# 创建动作噪声（用于TD3的探索）
n_actions = vec_env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

# 设置评估环境
eval_env = DummyVecEnv([lambda: gym.make(env_id)])


# 创建回调函数
# 定期评估模型性能
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{model_dir}/best_model",
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=10,
)

# 合并回调函数
callback_list = CallbackList([eval_callback])

# 创建并配置TD3模型
model = TD3(
    policy="MlpPolicy",              # 使用多层感知机策略网络
    env=vec_env,
    learning_rate=0.0003,            # 学习率
    buffer_size=100000,             # 经验回放缓冲区大小
    learning_starts=10000,           # 开始学习前收集的时间步数
    batch_size=256,                  # 每次梯度更新的批次大小
    tau=0.005,                       # 软更新目标网络的系数
    gamma=0.99,                      # 折扣因子
    train_freq=(1, "step"),          # 每个step更新一次
    gradient_steps=1,                # 每次更新执行的梯度下降步数
    action_noise=action_noise,       # 动作噪声
    policy_delay=2,                  # 策略网络更新频率（比Q网络慢）
    target_policy_noise=0.2,         # 目标策略平滑正则化的噪声
    target_noise_clip=0.5,           # 目标噪声的裁剪范围
    verbose=1,                       # 输出详细日志
    tensorboard_log=tensorboard_log, # TensorBoard日志目录
    device="auto",                   # 自动选择CPU或GPU
    seed=42                          # 随机种子
)

# 使用自定义日志记录器
model.set_logger(new_logger)

# 训练模型
total_timesteps = 1_000_000
model.learn(
    total_timesteps=total_timesteps,
    callback=callback_list,
    log_interval=1000,
    tb_log_name="td3_run",
    reset_num_timesteps=True,
    progress_bar=True
)

# 保存最终模型
final_model_path = f"{model_dir}/td3_halfcheetah_final"
model.save(final_model_path)

# # 加载并评估最终模型
# final_model_path = "./models/td3_halfcheetah_2025-03-25_21-24-28/td3_halfcheetah_final.zip"
# model_dir = "./models/td3_halfcheetah_2025-03-25_21-24-28/"
# loaded_model = TD3.load(final_model_path)
#
# # 确保评估环境使用训练环境的归一化参数
# env_id = "HalfCheetah-v5"
# eval_env = DummyVecEnv([lambda: gym.make(env_id)])  # 重新创建评估环境
# eval_env.training = False
# eval_env.norm_reward = False
# loaded_model.set_env(eval_env)
#
# # 评估模型（计算平均奖励）
# mean_reward, std_reward = evaluate_policy(
#     loaded_model,
#     eval_env,
#     n_eval_episodes=10,
#     deterministic=True
# )
#
# print(f"最终模型平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

