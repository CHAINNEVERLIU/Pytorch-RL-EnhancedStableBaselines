import gymnasium as gym
import numpy as np
from core import BCQ
from core.common.evaluation import evaluate_policy
from core.common.noise import NormalActionNoise
from core.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from core.common.monitor import Monitor
from core.common.vec_env import DummyVecEnv, VecNormalize
from core.common.logger import configure
from gymnasium.wrappers import RecordVideo
from core.common.save_util import load_from_pkl, save_to_pkl
import os
from datetime import datetime


timestamp = datetime.now().strftime("%Y-%m-%d_%H")
log_dir = f"./logs/bcq_halfcheetah"
model_dir = f"./models/bcq_halfcheetah"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 配置日志记录器
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# 创建向量化环境（支持并行训练多个环境实例）
env_id = "HalfCheetah-v5"
vec_env = DummyVecEnv([lambda: gym.make(env_id)])
vec_env.training = False  # 不更新统计数据
vec_env.norm_reward = False  # 不归一化奖励

# 应用向量化归一化包装器（对状态和奖励进行归一化）
offline_buffer_path = "./offline_data/td3_halfcheetah_expert_buffer.pkl"


# 创建回调函数,定期评估模型性能,设置评估环境
eval_env = DummyVecEnv([lambda: gym.make(env_id)])
eval_env.training = False
eval_env.norm_reward = False

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{model_dir}/best_model",
    log_path=log_dir,
    eval_freq=10_000,
    deterministic=True,
    render=False,
    n_eval_episodes=10,
)

# 合并回调函数
callback_list = CallbackList([eval_callback])


policy_kwargs = dict(
    actor_net_arch=dict(vae_latent_dim=12, vae_hidden_dim=700, perturbation_hidden_dim=400, max_perturbation=0.05),
    critic_net_arch=[400, 300],
)

model = BCQ(
        policy="MlpPolicy",
        env=vec_env,
        dataset=offline_buffer_path,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        gradient_steps=1,
        behavior_cloning_warmup=0,
        n_eval_episodes=10,
        policy_kwargs=policy_kwargs,
        stats_window_size=100,
        tensorboard_log=None,
        verbose=0,
        device="auto",
        seed=42,
        actor_delay=2,
)

# 使用自定义日志记录器
model.set_logger(new_logger)

# 训练模型
total_timesteps = 500_000
model.learn(
    total_timesteps=total_timesteps,
    callback=callback_list,
    log_interval=1000,
    tb_log_name="bcq_run",
    reset_num_timesteps=True,
    progress_bar=True
)

# 保存最终模型
final_model_path = f"{model_dir}/bcq_halfcheetah_final"
model.save(final_model_path)

# 评估模型（计算平均奖励）
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True
)

print(f"最终模型平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")


# # 加载并评估最终模型
# offline_buffer_path = "./offline_data/td3_halfcheetah_expert_buffer.pkl"
# final_model_path = "./models/bcq_halfcheetah/best_model/best_model.zip"
# model_dir = "./models/bcq_halfcheetah/"
# loaded_model = BCQ.load(final_model_path, dataset=offline_buffer_path)
#
# env_id = "HalfCheetah-v5"
# eval_env = DummyVecEnv([lambda: gym.make(env_id)])
# eval_env.training = False
# eval_env.norm_reward = False
# # 确保评估环境使用训练环境的归一化参数
# # 在实际评估前，将训练环境的归一化参数复制到评估环境
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

