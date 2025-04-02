import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from typing import List, Tuple, Dict, Union, Optional


class IndexedBox(gym.spaces.Box):
    """
    扩展 Box 空间，添加索引记录功能
    """

    def __init__(self, low, high, indices, dtype=np.float32):
        super().__init__(low=low, high=high, dtype=dtype)
        self.indices = np.array(indices)  # 保存原始空间中的索引

    def map_to_original(self, values):
        """将此子空间的值映射回原始空间中的适当位置"""
        if isinstance(values, list):
            values = np.array(values)
        assert values.shape == self.shape, f"值形状 {values.shape} 与空间形状 {self.shape} 不匹配"
        return self.indices, values


def split_spaces(
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        observation_splits: List[List[int]],
        action_splits: List[List[int]]
) -> Tuple[List[IndexedBox], List[IndexedBox]]:
    """
    根据提供的索引列表同时切分观测空间和动作空间，保留索引信息

    参数:
    - observation_space: 原始观测空间 (gym.spaces.Box)
    - action_space: 原始动作空间 (gym.spaces.Box)
    - observation_splits: 观测空间维度的索引列表，例如 [[0,1,2,3], [4,5,6,7], ...]
    - action_splits: 动作空间维度的索引列表，例如 [[0,1], [2,3], ...]

    返回:
    - 切分后的观测空间列表和动作空间列表（每个都包含原始索引）
    """
    obs_subspaces = []
    action_subspaces = []

    # 切分观测空间
    for indices in observation_splits:
        indices = np.array(indices)
        low = observation_space.low[indices]
        high = observation_space.high[indices]
        subspace = IndexedBox(low=low, high=high, indices=indices, dtype=observation_space.dtype)
        obs_subspaces.append(subspace)

    # 切分动作空间
    for indices in action_splits:
        indices = np.array(indices)
        low = action_space.low[indices]
        high = action_space.high[indices]
        subspace = IndexedBox(low=low, high=high, indices=indices, dtype=action_space.dtype)
        action_subspaces.append(subspace)

    return obs_subspaces, action_subspaces


class SubEnvironmentWrapper(gym.Wrapper):
    """
    创建子环境，只使用状态和动作空间的一部分，并记住索引
    """

    def __init__(
            self,
            env: gym.Env,
            obs_indices: List[int],
            action_indices: List[int],
            default_action: Optional[np.ndarray] = None,
            sub_env_id: int = None
    ):
        super().__init__(env)
        self.obs_indices = np.array(obs_indices)
        self.action_indices = np.array(action_indices)
        self.sub_env_id = sub_env_id  # 子环境ID

        # 设置观测空间
        if isinstance(env.observation_space, gym.spaces.Box):
            low = env.observation_space.low[self.obs_indices]
            high = env.observation_space.high[self.obs_indices]
            self.observation_space = IndexedBox(
                low=low, high=high, indices=self.obs_indices, dtype=env.observation_space.dtype
            )

        # 设置动作空间
        self.full_action_dim = env.action_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box):
            low = env.action_space.low[self.action_indices]
            high = env.action_space.high[self.action_indices]
            self.action_space = IndexedBox(
                low=low, high=high, indices=self.action_indices, dtype=env.action_space.dtype
            )

        # 设置默认动作值
        if default_action is None:
            self.default_action = np.zeros(self.full_action_dim)
        else:
            self.default_action = default_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self.obs_indices], info

    def step(self, action):
        # 构建完整的动作向量
        full_action = self.default_action.copy()
        full_action[self.action_indices] = action

        # 执行动作并只返回子集观测
        obs, reward, terminated, truncated, info = self.env.step(full_action)
        return obs[self.obs_indices], reward, terminated, truncated, info

    # 添加辅助方法，方便从子空间映射回原始空间
    def map_observation_to_original(self, sub_obs):
        """将子观测空间的观测映射回原始空间的相应位置"""
        return self.observation_space.map_to_original(sub_obs)

    def map_action_to_original(self, sub_action):
        """将子动作空间的动作映射回原始空间的相应位置"""
        return self.action_space.map_to_original(sub_action)

    def get_sub_env_info(self):
        """获取子环境的信息"""
        return {
            "sub_env_id": self.sub_env_id,
            "obs_indices": self.obs_indices.tolist(),
            "action_indices": self.action_indices.tolist(),
            "obs_space": self.observation_space,
            "action_space": self.action_space
        }


def split_environment(
        env: gym.Env,
        obs_splits: List[List[int]],
        action_splits: List[List[int]],
        default_actions: Optional[List[np.ndarray]] = None
) -> List[SubEnvironmentWrapper]:
    """
    根据提供的索引列表将环境切分成多个子环境

    参数:
    - env: 原始环境
    - obs_splits: 观测空间维度的索引列表
    - action_splits: 动作空间维度的索引列表
    - default_actions: 可选的默认动作值列表，用于未使用的动作维度

    返回:
    - 子环境包装器列表，每个都包含原始索引信息
    """
    assert len(obs_splits) == len(action_splits), "观测和动作的分割数量必须一致"

    if default_actions is None:
        default_actions = [None] * len(obs_splits)

    subenvs = []

    for i, (obs_idx, act_idx) in enumerate(zip(obs_splits, action_splits)):
        subenv = SubEnvironmentWrapper(
            env,
            obs_indices=obs_idx,
            action_indices=act_idx,
            default_action=default_actions[i],
            sub_env_id=i  # 分配子环境ID
        )
        subenvs.append(subenv)

    return subenvs


# 辅助函数 - 用于将多个子空间的动作组合回原始空间
def combine_actions(subenvs, sub_actions):
    """
    将多个子环境的动作组合成一个完整的动作向量

    参数:
    - subenvs: 子环境列表
    - sub_actions: 对应的子动作列表

    返回:
    - 完整的动作向量
    """
    # 获取完整动作空间维度
    full_action_dim = subenvs[0].full_action_dim
    full_action = np.zeros(full_action_dim)

    for subenv, sub_action in zip(subenvs, sub_actions):
        indices, values = subenv.map_action_to_original(sub_action)
        full_action[indices] = values

    return full_action


import gymnasium as gym
import numpy as np
from typing import List, Tuple, Dict, Union, Optional


class IndexedBox(gym.spaces.Box):
    """
    扩展 Box 空间，添加索引记录功能
    """

    def __init__(self, low, high, indices, dtype=np.float32):
        super().__init__(low=low, high=high, dtype=dtype)
        self.indices = np.array(indices)  # 保存原始空间中的索引

    def map_to_original(self, values):
        """将此子空间的值映射回原始空间中的适当位置"""
        if isinstance(values, list):
            values = np.array(values)
        assert values.shape == self.shape, f"值形状 {values.shape} 与空间形状 {self.shape} 不匹配"
        return self.indices, values


def split_spaces(
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        observation_splits: List[List[int]],
        action_splits: List[List[int]]
) -> Tuple[List[IndexedBox], List[IndexedBox]]:
    """
    根据提供的索引列表同时切分观测空间和动作空间，保留索引信息

    参数:
    - observation_space: 原始观测空间 (gym.spaces.Box)
    - action_space: 原始动作空间 (gym.spaces.Box)
    - observation_splits: 观测空间维度的索引列表，例如 [[0,1,2,3], [4,5,6,7], ...]
    - action_splits: 动作空间维度的索引列表，例如 [[0,1], [2,3], ...]

    返回:
    - 切分后的观测空间列表和动作空间列表（每个都包含原始索引）
    """
    obs_subspaces = []
    action_subspaces = []

    # 切分观测空间
    for indices in observation_splits:
        indices = np.array(indices)
        low = observation_space.low[indices]
        high = observation_space.high[indices]
        subspace = IndexedBox(low=low, high=high, indices=indices, dtype=observation_space.dtype)
        obs_subspaces.append(subspace)

    # 切分动作空间
    for indices in action_splits:
        indices = np.array(indices)
        low = action_space.low[indices]
        high = action_space.high[indices]
        subspace = IndexedBox(low=low, high=high, indices=indices, dtype=action_space.dtype)
        action_subspaces.append(subspace)

    return obs_subspaces, action_subspaces


class SubEnvironmentWrapper(gym.Wrapper):
    """
    创建子环境，只使用状态和动作空间的一部分，并记住索引
    """

    def __init__(
            self,
            env: gym.Env,
            obs_indices: List[int],
            action_indices: List[int],
            default_action: Optional[np.ndarray] = None,
            sub_env_id: int = None
    ):
        super().__init__(env)
        self.obs_indices = np.array(obs_indices)
        self.action_indices = np.array(action_indices)
        self.sub_env_id = sub_env_id  # 子环境ID

        # 设置观测空间
        if isinstance(env.observation_space, gym.spaces.Box):
            low = env.observation_space.low[self.obs_indices]
            high = env.observation_space.high[self.obs_indices]
            self.observation_space = IndexedBox(
                low=low, high=high, indices=self.obs_indices, dtype=env.observation_space.dtype
            )

        # 设置动作空间
        self.full_action_dim = env.action_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box):
            low = env.action_space.low[self.action_indices]
            high = env.action_space.high[self.action_indices]
            self.action_space = IndexedBox(
                low=low, high=high, indices=self.action_indices, dtype=env.action_space.dtype
            )

        # 设置默认动作值
        if default_action is None:
            self.default_action = np.zeros(self.full_action_dim)
        else:
            self.default_action = default_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self.obs_indices], info

    def step(self, action):
        # 构建完整的动作向量
        full_action = self.default_action.copy()
        full_action[self.action_indices] = action

        # 执行动作并只返回子集观测
        obs, reward, terminated, truncated, info = self.env.step(full_action)
        return obs[self.obs_indices], reward, terminated, truncated, info

    # 添加辅助方法，方便从子空间映射回原始空间
    def map_observation_to_original(self, sub_obs):
        """将子观测空间的观测映射回原始空间的相应位置"""
        return self.observation_space.map_to_original(sub_obs)

    def map_action_to_original(self, sub_action):
        """将子动作空间的动作映射回原始空间的相应位置"""
        return self.action_space.map_to_original(sub_action)

    def get_sub_env_info(self):
        """获取子环境的信息"""
        return {
            "sub_env_id": self.sub_env_id,
            "obs_indices": self.obs_indices.tolist(),
            "action_indices": self.action_indices.tolist(),
            "obs_space": self.observation_space,
            "action_space": self.action_space
        }


def split_environment(
        env: gym.Env,
        obs_splits: List[List[int]],
        action_splits: List[List[int]],
        default_actions: Optional[List[np.ndarray]] = None
) -> List[SubEnvironmentWrapper]:
    """
    根据提供的索引列表将环境切分成多个子环境

    参数:
    - env: 原始环境
    - obs_splits: 观测空间维度的索引列表
    - action_splits: 动作空间维度的索引列表
    - default_actions: 可选的默认动作值列表，用于未使用的动作维度

    返回:
    - 子环境包装器列表，每个都包含原始索引信息
    """
    assert len(obs_splits) == len(action_splits), "观测和动作的分割数量必须一致"

    if default_actions is None:
        default_actions = [None] * len(obs_splits)

    subenvs = []

    for i, (obs_idx, act_idx) in enumerate(zip(obs_splits, action_splits)):
        subenv = SubEnvironmentWrapper(
            env,
            obs_indices=obs_idx,
            action_indices=act_idx,
            default_action=default_actions[i],
            sub_env_id=i  # 分配子环境ID
        )
        subenvs.append(subenv)

    return subenvs


# ------ 添加空间重组功能 ------

def combine_spaces_from_indices(
        original_obs_space: gym.spaces.Box,
        original_action_space: gym.spaces.Box,
        obs_indices_list: List[List[int]],
        action_indices_list: List[List[int]]
) -> Tuple[gym.spaces.Box, gym.spaces.Box]:
    """
    根据索引列表重新组合原始空间

    参数:
    - original_obs_space: 原始观测空间
    - original_action_space: 原始动作空间
    - obs_indices_list: 观测空间索引列表
    - action_indices_list: 动作空间索引列表

    返回:
    - 重组后的观测空间和动作空间
    """
    # 确保所有索引都被覆盖
    all_obs_indices = np.concatenate(obs_indices_list)
    all_action_indices = np.concatenate(action_indices_list)

    # 检查是否有重复索引
    if len(set(all_obs_indices)) != len(all_obs_indices):
        raise ValueError("观测空间索引中存在重复")
    if len(set(all_action_indices)) != len(all_action_indices):
        raise ValueError("动作空间索引中存在重复")

    # 创建新的空间
    return original_obs_space, original_action_space


def combine_spaces(
        subspaces: List[IndexedBox],
        original_space: gym.spaces.Box
) -> gym.spaces.Box:
    """
    将多个子空间组合回完整空间

    参数:
    - subspaces: IndexedBox子空间列表
    - original_space: 原始完整空间模板

    返回:
    - 组合后的完整空间
    """
    # 验证所有子空间索引
    all_indices = []
    for space in subspaces:
        all_indices.extend(space.indices)

    # 检查是否有重复或遗漏
    unique_indices = set(all_indices)
    if len(unique_indices) != len(all_indices):
        raise ValueError("子空间索引中存在重复")

    # 检查索引范围是否超出原始空间
    if max(all_indices) >= original_space.shape[0]:
        raise ValueError(f"索引 {max(all_indices)} 超出原始空间范围 {original_space.shape[0]}")

    # 创建新空间（这里简单返回原始空间，因为我们已经确认了索引覆盖）
    return original_space


def combine_values(
        subspaces: List[IndexedBox],
        values_list: List[np.ndarray],
        full_dim: int
) -> np.ndarray:
    """
    将多个子空间的值组合成一个完整的向量

    参数:
    - subspaces: 子空间列表
    - values_list: 对应的值列表
    - full_dim: 完整向量的维度

    返回:
    - 组合后的完整向量
    """
    full_values = np.zeros(full_dim)

    for space, values in zip(subspaces, values_list):
        indices, sub_values = space.map_to_original(values)
        full_values[indices] = sub_values

    return full_values


def combine_observations(
        subspaces: List[IndexedBox],
        observations: List[np.ndarray],
        full_obs_dim: int
) -> np.ndarray:
    """
    将多个子空间的观测组合成一个完整的观测向量
    """
    return combine_values(subspaces, observations, full_obs_dim)


def combine_actions(
        subspaces: List[IndexedBox],
        actions: List[np.ndarray],
        full_action_dim: int
) -> np.ndarray:
    """
    将多个子空间的动作组合成一个完整的动作向量
    """
    return combine_values(subspaces, actions, full_action_dim)


def combine_subenvs_actions(
        subenvs: List[SubEnvironmentWrapper],
        sub_actions: List[np.ndarray]
) -> np.ndarray:
    """
    将多个子环境的动作组合成一个完整的动作向量

    参数:
    - subenvs: 子环境列表
    - sub_actions: 对应的子动作列表

    返回:
    - 完整的动作向量
    """
    # 获取完整动作空间维度
    full_action_dim = subenvs[0].full_action_dim
    full_action = np.zeros(full_action_dim)

    for subenv, sub_action in zip(subenvs, sub_actions):
        indices, values = subenv.map_action_to_original(sub_action)
        full_action[indices] = values

    return full_action


# 示例使用
def example_usage():
    # 创建一个环境，假设有16维状态空间和8维动作空间
    env = gym.make('Humanoid-v4')  # 这只是一个例子

    # 将状态空间分成4个子空间，每个4维
    obs_splits = [
        list(range(0, 4)),
        list(range(4, 8)),
        list(range(8, 12)),
        list(range(12, 16))
    ]

    # 将动作空间分成4个子空间，每个2维
    action_splits = [
        list(range(0, 2)),
        list(range(2, 4)),
        list(range(4, 6)),
        list(range(6, 8))
    ]

    # 1. 只获取切分后的空间表示
    obs_subspaces, action_subspaces = split_spaces(
        env.observation_space, env.action_space, obs_splits, action_splits
    )

    # 检查每个子空间是否包含索引信息
    for i, (obs_space, act_space) in enumerate(zip(obs_subspaces, action_subspaces)):
        print(f"子空间 {i}:")
        print(f"  观测空间索引: {obs_space.indices}")
        print(f"  动作空间索引: {act_space.indices}")

    # 2. 创建子环境
    subenvs = split_environment(env, obs_splits, action_splits)

    # 使用子环境并访问索引信息
    for i, subenv in enumerate(subenvs):
        info = subenv.get_sub_env_info()
        print(f"子环境 {info['sub_env_id']}:")
        print(f"  观测空间索引: {info['obs_indices']}")
        print(f"  动作空间索引: {info['action_indices']}")

    # 示例：如何将多个子环境的动作组合回原始空间
    # 假设我们有来自不同子环境的动作
    sub_actions = [
        np.array([0.1, 0.2]),  # 子环境0的动作
        np.array([0.3, 0.4]),  # 子环境1的动作
        np.array([0.5, 0.6]),  # 子环境2的动作
        np.array([0.7, 0.8])  # 子环境3的动作
    ]

    # 组合成完整动作
    full_action = combine_actions(subenvs, sub_actions)
    print(f"组合后的完整动作: {full_action}")

    return obs_subspaces, action_subspaces, subenvs








