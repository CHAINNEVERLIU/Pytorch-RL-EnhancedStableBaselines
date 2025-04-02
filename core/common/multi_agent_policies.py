import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from core.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from core.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from core.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from core.common.type_aliases import PyTorchObs, Schedule
from core.common.utils import get_device, is_vectorized_observation, obs_as_tensor

from core.common.policies import BasePolicy


SelfMultiAgentBaseModel = TypeVar("SelfMultiAgentBaseModel", bound="MultiAgentBaseModel")


class MultiAgentBaseModel(nn.Module):
    """
    Multi-agent policy base class. The core concept is to store all agent policies
    in a single container class, so one multi-agent policy class contains policies
    for all agents in the environment.
    """
    optimizer_list: list[th.optim.Optimizer]

    def __init__(
            self,
            n_agents: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            observation_space_list: list[spaces.Space],
            action_space_list: list[spaces.Space],
            features_extractor_class_list: list[type[BaseFeaturesExtractor]] = None,
            features_extractor_kwargs_list: Optional[list[dict[str, Any]]] = None,
            features_extractor_list: Optional[list[BaseFeaturesExtractor]] = None,
            normalize_images: bool = True,
            optimizer_class_list: list[type[th.optim.Optimizer]] = None,
            optimizer_kwargs_list: Optional[list[dict[str, Any]]] = None,
    ):
        """
        Initialize the multi-agent model.

        :param n_agents: Number of agents
        :param observation_space: Combined observation space for all agents
        :param action_space: Combined action space for all agents
        :param observation_space_list: List of individual observation spaces
        :param action_space_list: List of individual action spaces
        :param features_extractor_class_list: List of feature extractor classes, one per agent
        :param features_extractor_kwargs_list: List of feature extractor kwargs, one per agent
        :param features_extractor_list: Optional pre-created feature extractors
        :param normalize_images: Whether to normalize images
        :param optimizer_class_list: List of optimizer classes, one per agent
        :param optimizer_kwargs_list: List of optimizer kwargs, one per agent
        """
        super(MultiAgentBaseModel, self).__init__()
        # Initialize default values if not provided
        if features_extractor_class_list is None:
            features_extractor_class_list = [FlattenExtractor for _ in range(n_agents)]

        if features_extractor_kwargs_list is None:
            features_extractor_kwargs_list = [{} for _ in range(n_agents)]

        if optimizer_class_list is None:
            optimizer_class_list = [th.optim.Adam for _ in range(n_agents)]

        if optimizer_kwargs_list is None:
            optimizer_kwargs_list = [{} for _ in range(n_agents)]

        # Check list lengths are consistent
        if not (n_agents == len(features_extractor_class_list)
                == len(features_extractor_kwargs_list) == len(optimizer_class_list)
                == len(optimizer_kwargs_list)):
            raise ValueError(
                f"In a multi-agent scenario, the number of [n_agents, len(features_extractor_class_list), "
                f"len(features_extractor_kwargs_list), len(optimizer_class_list), len(optimizer_kwargs_list)"
                f"] must be consistent, now they are [{n_agents}, {len(features_extractor_class_list)}, "
                f"{len(features_extractor_kwargs_list)}, {len(optimizer_class_list)}, {len(optimizer_kwargs_list)}].")

        self.n_agents = n_agents
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_space_list = observation_space_list
        self.action_space_list = action_space_list
        self.features_extractor_list = features_extractor_list
        self.normalize_images = normalize_images

        self.optimizer_class_list = optimizer_class_list
        self.optimizer_kwargs_list = optimizer_kwargs_list
        # Initialize optimizer list to be populated later
        self.optimizer_list = []

        self.features_extractor_class_list = features_extractor_class_list
        self.features_extractor_kwargs_list = features_extractor_kwargs_list

        # Adjust kwargs for image normalization if needed
        for agent_id, features_extractor_class in enumerate(self.features_extractor_class_list):
            # Automatically deactivate dtype and bounds checks
            if not normalize_images and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
                self.features_extractor_kwargs_list[agent_id].update(dict(normalized_image=True))

    def _update_features_extractors(
        self,
        net_kwargs: dict[str, Any],
        features_extractor_list: Optional[list[BaseFeaturesExtractor]] = None,
    ) -> dict[str, Any]:
        """
         Update or create feature extractors for all agents.

        :param net_kwargs: Network keyword arguments including n_agents
        :param features_extractor_list: List of feature extractors, if None, new ones will be created
        :return: Updated network kwargs with features_extractor_list and features_dim_list
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor_list is None:
            features_extractor_list = [None for _ in range(net_kwargs["n_agents"])]

        assert net_kwargs["n_agents"] == len(features_extractor_list), \
            f"Expecting n_agents and features_extractor_list to be of the same length"

        features_extractors = []
        features_dim_list = []
        for agent_id, (features_extractor) in enumerate(features_extractor_list):
            if features_extractor is None:
                # The features extractor is not shared, create a new one
                features_extractor = self.agent_make_features_extractor(agent_id)

            features_extractors.append(features_extractor)
            features_dim_list.append(features_extractor.features_dim)

        net_kwargs.update(dict(features_extractor_list=features_extractors,
                               features_dim_list=features_dim_list))

        return net_kwargs

    def agent_make_features_extractor(self, agent_id) -> BaseFeaturesExtractor:
        """
        Create a features extractor for a specific agent.

        :param agent_id: Agent identifier
        :return: Features extractor instance for the specified agent
        """
        return self.features_extractor_class_list[agent_id](self.observation_space_list[agent_id],
                                                            **self.features_extractor_kwargs_list[agent_id])

    def agent_extract_features(self, agent_id: int, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor)\
            -> th.Tensor:
        """
        Preprocess the observation and extract features for a specific agent.

        :param agent_id: Agent identifier
        :param obs: Observation for the agent
        :param features_extractor: Features extractor to use
        :return: Extracted features as tensor
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space_list[agent_id],
                                          normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def init_optimizers(self, parameters_list=None, lr_schedule_list=None):
        """
        Initialize optimizers for all agents.

        :param parameters_list: Optional list of parameters to optimize per agent,
                               if None, will try to use agent networks' parameters
        :param lr_schedule_list:
        """
        self.optimizer_list = []

        # If no parameters provided, need to determine parameters from agent networks
        # This implementation will depend on the specific child class structure
        if parameters_list is None:
            # Example implementation - override in child classes based on their network structure
            raise NotImplementedError(
                "When parameters_list is None, child classes must override this method to provide parameters")

        for agent_id in range(self.n_agents):
            optimizer = self.optimizer_class_list[agent_id](
                parameters_list[agent_id],
                lr=lr_schedule_list[agent_id](1),
                **self.optimizer_kwargs_list[agent_id]
            )
            self.optimizer_list.append(optimizer)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            n_agents=self.n_agents,
            observation_space=self.observation_space,
            action_space=self.action_space,
            observation_space_list=self.observation_space_list,
            action_space_list=self.action_space_list,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            # print("ceshi:", param.device, param)
            return param.device
        return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls: type[SelfMultiAgentBaseModel], path: str, device: Union[th.device, str] = "auto")\
            -> SelfMultiAgentBaseModel:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        # Note(antonin): we cannot use `weights_only=True` here because we need to allow
        # gymnasium imports for the policy to be loaded successfully
        saved_variables = th.load(path, map_location=device, weights_only=False)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        th.nn.utils.vector_to_parameters(th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def agent_is_vectorized_observation(self, agent_id: int,
                                        observation: Union[np.ndarray, dict[str, np.ndarray]]) -> bool:
        """
        Check whether the observation for a specific agent is vectorized.
        Apply transposition to images (channel-first) if needed.

        :param agent_id: Agent identifier
        :param observation: Observation to check
        :return: True if the observation is vectorized, False otherwise
        """
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(
                self.observation_space_list[agent_id], spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.observation_space_list[agent_id]}"
            for key, obs in observation.items():
                obs_space = self.observation_space_list[agent_id].spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space_list[agent_id]), self.observation_space_list[agent_id]
            )
        return vectorized_env

    def obs_to_tensor(self, observation: Union[np.ndarray, dict[str, np.ndarray]]) \
            -> tuple[PyTorchObs, bool]:
        """
        Convert combined observation for all agents to PyTorch tensor.

        :param observation: Combined observation for all agents
        :return: Tuple of (observation tensor, whether observation is vectorized)
        """
        vectorized_env = False
        if isinstance(observation, dict):
            raise ValueError("Dictionary observation spaces are not currently supported")
        elif is_image_space(self.observation_space):
            raise ValueError("Image observation spaces are not currently supported")
        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            # 这个地方有点小bug，所以直接返回true
            vectorized_env = True
            # vectorized_env = is_vectorized_observation(observation, self.observation_space_list[agent_id])

            # Add batch dimension if needed
            observation = observation.reshape(-1, *self.observation_space.shape)  # type: ignore[misc]
        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env

    def agent_obs_to_tensor(self, agent_id: int, observation: Union[np.ndarray, dict[str, np.ndarray]]) \
            -> tuple[PyTorchObs, bool]:
        """
        Convert agent-specific observation to PyTorch tensor.

        :param agent_id: Agent identifier
        :param observation: Observation for the specific agent
        :return: Tuple of (observation tensor, whether observation is vectorized)
        """
        vectorized_env = False
        if isinstance(observation, dict):
            raise ValueError("Dictionary observation spaces are not currently supported")
        elif is_image_space(self.observation_space_list[agent_id]):
            raise ValueError("Image observation spaces are not currently supported")
        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            # 这个地方有点小bug，所以直接返回true
            vectorized_env = True
            # vectorized_env = is_vectorized_observation(observation, self.observation_space_list[agent_id])

            # Add batch dimension if needed
            observation = observation.reshape(-1, *self.observation_space_list[agent_id].shape)  # type: ignore[misc]
        obs_tensor = obs_as_tensor(observation, self.device)
        return obs_tensor, vectorized_env

    def _agent_obs_tensor_extract(
            self,
            agent_id: int,
            global_observation: Union[th.Tensor, dict[str, th.Tensor]],
    ) -> Union[th.Tensor, Optional[tuple[th.Tensor, ...]]]:
        """
        从全局观察张量中提取特定智能体的观察张量。

        :param agent_id: 智能体ID
        :param global_observation: 全局观察张量
        :return: 提取的该智能体的观察张量
        """
        # 获取该智能体观察空间对应的索引
        # 注意：这里假设observation_space_list中的每个空间都有indices属性
        # 如果不是所有空间都有这个属性，需要修改此处逻辑
        if hasattr(self.observation_space_list[agent_id], 'indices'):
            agent_indices = self.observation_space_list[agent_id].indices
            agent_obs = global_observation[..., agent_indices]
        else:
            # 如果没有indices属性，可以提供一个替代方案
            # 例如，假设观察空间是按顺序排列的
            # 这里需要根据实际情况进行修改
            raise NotImplementedError(
                "observation_space_list[agent_id] does not have 'indices' attribute. "
                "You need to implement a custom extraction method."
            )
        return agent_obs

    def _agent_action_tensor_extract(
            self,
            agent_id: int,
            global_action: Union[th.Tensor, dict[str, th.Tensor]],
    ) -> Union[th.Tensor, Optional[tuple[th.Tensor, ...]]]:
        """
        从全局动作张量中提取特定智能体的动作张量。

        :param agent_id: 智能体ID
        :param global_action: 全局动作张量
        :return: 提取的该智能体的动作张量
        """
        # 获取该智能体动作空间对应的索引
        # 注意：这里假设action_space_list中的每个空间都有indices属性
        if hasattr(self.action_space_list[agent_id], 'indices'):
            agent_indices = self.action_space_list[agent_id].indices
            agent_action = global_action[..., agent_indices]
        else:
            # 如果没有indices属性，可以提供一个替代方案
            raise NotImplementedError(
                "action_space_list[agent_id] does not have 'indices' attribute. "
                "You need to implement a custom extraction method."
            )
        return agent_action


class MultiAgentBasePolicy(MultiAgentBaseModel, ABC):
    """
    所有多智能体策略 (Policy) 的基类。

    该类是一个抽象类，子类必须实现 `_agent_predict` 方法。

    :param args: 传递给 `MultiAgentBaseModel` 的位置参数。
    :param kwargs: 传递给 `MultiAgentBaseModel` 的关键字参数。
    :param squash_output: 对于连续动作空间，是否使用 `tanh()` 函数对输出进行压缩。
    """

    features_extractor_list: list[BaseFeaturesExtractor]

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """
        提供一个恒定值的调度器（用于 pickle 兼容性）。
        :param progress_remaining: 训练剩余进度（该方法不使用此参数）。
        :return: 固定学习率 `0.0`
        """
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        使用正交初始化对神经网络层进行初始化。

        :param module: 需要初始化的神经网络层。
        :param gain: 初始化增益值。
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _agent_predict(self, agent_id: int, agent_observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        计算给定智能体观测值的策略动作（抽象方法，子类必须实现）。

        :param agent_id: 智能体ID
        :param agent_observation: PyTorch 张量格式的该智能体的观察值
        :param deterministic: 是否使用确定性策略
        :return: 计算出的动作张量
        """
        pass

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        计算全局观测值的策略动作，通过调用每个智能体的预测方法并合并结果。

        :param observation: PyTorch 张量格式的全局观察值
        :param deterministic: 是否使用确定性策略
        :return: 合并后的全局动作张量
        """
        actions = []
        for agent_id in range(self.n_agents):
            # 从全局观察中提取该智能体的观察
            agent_obs_tensor = self._agent_obs_tensor_extract(agent_id, observation)
            # 预测该智能体的动作
            action = self._agent_predict(agent_id, agent_obs_tensor, deterministic=deterministic)
            actions.append(action)

        # 合并所有智能体的动作
        actions = th.cat(actions, dim=-1)
        return actions

    def predict(
            self,
            observation: Union[np.ndarray, dict[str, np.ndarray]],
            state: Optional[tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        计算给定观察值的动作。

        :param observation: 观察值（可以是 NumPy 数组或字典）。
        :param state: 过去的隐藏状态（适用于 RNN 策略）。
        :param episode_start: 是否是新 episode（适用于 RNN 策略）。
        :param deterministic: 是否使用确定性策略。
        :return: 计算出的动作以及新的隐藏状态（如果适用）。
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # 确保用户没有混用 Gym API 和 SB3 VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        # 方法1：先转换全局观察，再分别提取每个智能体的观察
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        global_actions = []
        for agent_id in range(self.n_agents):
            # 提取该智能体的观察张量
            agent_obs_tensor = self._agent_obs_tensor_extract(agent_id, obs_tensor)

            # 使用no_grad避免计算梯度
            with th.no_grad():
                actions = self._agent_predict(agent_id, agent_obs_tensor, deterministic=deterministic)

            # 转换为numpy数组，并重塑为原始动作形状
            actions = actions.cpu().numpy().reshape((-1, *self.action_space_list[agent_id].shape))

            # 处理连续动作空间
            if isinstance(self.action_space_list[agent_id], spaces.Box):
                if self.squash_output:
                    # 如果使用squashing，将动作重新缩放到正确的域
                    actions = self.agent_unscale_action(agent_id, actions)
                else:
                    # 动作可能在任意尺度上，所以裁剪动作以避免越界错误
                    actions = np.clip(actions, self.action_space_list[agent_id].low,
                                      self.action_space_list[agent_id].high)

            # 如果需要，移除批处理维度
            if not vectorized_env:
                assert isinstance(actions, np.ndarray)
                actions = actions.squeeze(axis=0)

            global_actions.append(actions)

        global_actions = np.concatenate(global_actions, axis=-1)
        return global_actions, state

    def agent_scale_action(self, agent_id: int, action: np.ndarray) -> np.ndarray:
        """
        将动作值从原始范围缩放到[-1, 1]。

        :param agent_id: 智能体ID
        :param action: 原始动作值
        :return: 缩放后的动作值
        """
        assert isinstance(
            self.action_space_list[agent_id], spaces.Box
        ), f"Trying to scale an action using an action space that is not a Box(): {self.action_space_list[agent_id]}"
        low, high = self.action_space_list[agent_id].low, self.action_space_list[agent_id].high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def agent_unscale_action(self, agent_id: int, scaled_action: np.ndarray) -> np.ndarray:
        """
        将[-1, 1]范围内的缩放动作值恢复到原始范围。

        :param agent_id: 智能体ID
        :param scaled_action: 缩放后的动作值
        :return: 恢复到原始范围的动作值
        """
        assert isinstance(
            self.action_space_list[agent_id], spaces.Box
        ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space_list[agent_id]}"
        low, high = self.action_space_list[agent_id].low, self.action_space_list[agent_id].high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        将动作值从原始范围缩放到[-1, 1]。

        :param action: 原始动作值
        :return: 缩放后的动作值
        """
        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to scale an action using an action space that is not a Box(): {self.action_space}"
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        将[-1, 1]范围内的缩放动作值恢复到原始范围。

        :param scaled_action: 缩放后的动作值
        :return: 恢复到原始范围的动作值
        """
        assert isinstance(
            self.action_space, spaces.Box
        ), f"Trying to unscale an action using an action space that is not a Box(): {self.action_space}"
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """
        获取重建模型所需的数据。

        :return: 用于重建模型的构造函数参数字典
        """
        data = super()._get_constructor_parameters()
        data.update(dict(squash_output=self.squash_output))
        return data

    def reset_noise(self, num_envs):
        """
        一些可以使用状态依赖的Actor需要重构此类
        :param num_envs:
        :return:
        """
        pass










