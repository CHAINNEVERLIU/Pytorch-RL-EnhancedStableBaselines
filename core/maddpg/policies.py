from typing import Any, Optional, Union

import torch as th
from gymnasium import spaces
from torch import nn

from core.common.multi_agent_policies import MultiAgentBasePolicy, MultiAgentBaseModel
from core.common.preprocessing import get_action_dim
from core.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_multi_agent_actor_critic_arch,
)
from core.common.type_aliases import PyTorchObs, Schedule
from core.common.envs.multi_agent_envs import combine_spaces


class Actor(MultiAgentBasePolicy):
    """
    Actor network (policy) for MADDPG.

    :param n_agents: Number of agents
    :param observation_space: Combined observation space for all agents
    :param action_space: Combined action space for all agents
    :param observation_space_list: List of individual observation spaces
    :param action_space_list: List of individual action spaces
    :param net_arch: List of neural network architectures for each agent
    :param features_extractor_list: List of feature extractors for each agent
    :param features_dim_list: List of feature dimensions for each agent
    :param activation_fn: Activation function used in the neural networks
    :param normalize_images: Whether to normalize images
    """
    def __init__(
            self,
            n_agents: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            observation_space_list: list[spaces.Space],
            action_space_list: list[spaces.Space],
            net_arch: list[list[int]],
            features_extractor_list: list[nn.Module],
            features_dim_list: list[int],
            activation_fn: type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            n_agents=n_agents,
            observation_space=observation_space,
            action_space=action_space,
            observation_space_list=observation_space_list,
            action_space_list=action_space_list,
            features_extractor_list=features_extractor_list,
            normalize_images=normalize_images,
            squash_output=True,
        )
        self.net_arch = net_arch
        self.features_dim_list = features_dim_list
        self.activation_fn = activation_fn

        mu_list = nn.ModuleList([])
        for agent_id, action_space in enumerate(self.action_space_list):
            action_dim = get_action_dim(action_space)
            # 创建MLP网络将特征映射到动作
            actor_net = create_mlp(features_dim_list[agent_id], action_dim, net_arch[agent_id],
                                   activation_fn, squash_output=True)
            # 确定性动作输出
            mu = nn.Sequential(*actor_net)
            mu_list.append(mu)
        self.mu_list = mu_list

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """获取构造函数参数，用于模型保存和加载"""
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim_list=self.features_dim_list,
                activation_fn=self.activation_fn,
                features_extractor_list=self.features_extractor_list,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        前向传播方法，生成所有智能体的确定性动作。

        :param obs: 全局观察张量，形状为(batch_size, obs_dim)
        :return: 所有智能体合并的动作张量，形状为(batch_size, total_action_dim)
        """
        # MADDPG Actor只输出确定性动作
        global_actions = []
        for agent_id in range(self.n_agents):
            # 从全局观察提取当前智能体的观察
            agent_obs = self._agent_obs_tensor_extract(agent_id, obs)
            # 提取观察的特征
            features = self.agent_extract_features(agent_id, agent_obs, self.features_extractor_list[agent_id])
            # 生成当前智能体的动作
            agent_action = self.mu_list[agent_id](features)
            global_actions.append(agent_action)
        # 合并所有智能体的动作
        actions = th.cat(global_actions, dim=-1)
        return actions

    def _agent_predict(self, agent_id: int, agent_observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        为特定智能体预测动作。

        :param agent_id: 智能体ID
        :param agent_observation: 智能体的观察张量
        :param deterministic: 是否使用确定性策略（MADDPG总是确定性的）
        :return: 智能体的动作张量
        """
        # 注意：在MADDPG中，deterministic参数被忽略，因为预测总是确定性的
        features = self.agent_extract_features(agent_id, agent_observation, self.features_extractor_list[agent_id])
        agent_action = self.mu_list[agent_id](features)
        return agent_action


class ContinuousCritic(MultiAgentBaseModel):
    """
    连续动作空间的Critic网络，用于MADDPG算法。

    在MADDPG中，每个智能体的Critic评估所有智能体的联合状态和动作。
    实现了双Q网络结构，可以减少Q值估计的过高估计。

    :param n_agents: 智能体数量
    :param observation_space: 所有智能体的组合观察空间
    :param action_space: 所有智能体的组合动作空间
    :param observation_space_list: 每个智能体的观察空间列表
    :param action_space_list: 每个智能体的动作空间列表
    :param net_arch: 每个智能体的网络结构列表
    :param features_extractor_list: 每个智能体的特征提取器列表
    :param features_dim_list: 每个智能体的特征维度列表
    :param activation_fn: 激活函数类型，默认为ReLU
    :param normalize_images: 是否归一化图像，默认为True
    :param n_critics: 每个智能体的Q网络数量，默认为2（双Q网络）
    :param share_features_extractor: 是否与Actor共享特征提取器，默认为True
    """
    features_extractor_list: list[BaseFeaturesExtractor]

    def __init__(
        self,
        n_agents: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        observation_space_list: list[spaces.Space],
        action_space_list: list[spaces.Box],
        net_arch: list[list[int]],
        features_extractor_list: list[BaseFeaturesExtractor],
        features_dim_list: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            n_agents=n_agents,
            observation_space=observation_space,
            action_space=action_space,
            observation_space_list=observation_space_list,
            action_space_list=action_space_list,
            features_extractor_list=features_extractor_list,
            normalize_images=normalize_images)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.net_arch = net_arch
        self.features_dim_list = features_dim_list
        self.activation_fn = activation_fn

        # 创建Q网络
        self.q_networks_list: list[list[nn.Module]] = []

        # 计算特征和动作的总维度，用于Q网络的输入
        features_dim_sum = sum(features_dim_list)
        action_dim_sum = 0
        for agent_id, action_space in enumerate(self.action_space_list):
            action_dim_sum += get_action_dim(action_space)

        # 为每个智能体创建n_critics个Q网络
        for agent_id in range(self.n_agents):
            q_networks: list[nn.Module] = []
            for idx in range(n_critics):
                # 创建Q网络，输入是所有智能体的联合特征和动作
                q_net_list = create_mlp(features_dim_sum + action_dim_sum, 1, net_arch[agent_id], activation_fn)
                q_net = nn.Sequential(*q_net_list)
                # 为模块命名，便于调试和保存/加载
                self.add_module(f"agent{agent_id}_qf{idx}", q_net)
                q_networks.append(q_net)
            self.q_networks_list.append(q_networks)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """获取构造函数参数，用于模型保存和加载"""
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim_list=self.features_dim_list,
                activation_fn=self.activation_fn,
                n_critics=self.n_critics,
                share_features_extractor=self.share_features_extractor,
                features_extractor_list=self.features_extractor_list,
            )
        )
        return data

    def extract_features(self, agent_id: int, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
        """
        使用特征提取器从观察中提取特征。

        :param agent_id: 智能体ID
        :param obs: 观察张量
        :param features_extractor: 特征提取器
        :return: 提取的特征张量
        """
        return self.agent_extract_features(agent_id, obs, features_extractor)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> list[tuple[th.Tensor, ...]]:
        """
        前向传播方法，计算每个智能体的Q值。

        :param obs: 全局观察张量，形状为(batch_size, obs_dim)
        :param actions: 全局动作张量，形状为(batch_size, total_action_dim)
        :return: 每个智能体的Q值元组列表，每个元组包含n_critics个Q值
        """
        q_value_list = []
        features = []

        # 提取每个智能体的特征并合并
        for agent_id in range(self.n_agents):
            # 当特征提取器与Actor共享时，仅使用策略损失学习特征提取器
            with th.set_grad_enabled(not self.share_features_extractor):
                agent_obs = self._agent_obs_tensor_extract(agent_id, obs)
                agent_features = self.extract_features(agent_id, agent_obs, self.features_extractor_list[agent_id])
            features.append(agent_features)

        # 合并所有特征和动作作为Q网络的输入
        features = th.cat(features, dim=-1)
        qvalue_input = th.cat([features, actions], dim=1)

        for agent_id in range(self.n_agents):
            agent_q_values = tuple(q_net(qvalue_input) for q_net in self.q_networks_list[agent_id])
            q_value_list.append(agent_q_values)

        return q_value_list

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> list[th.Tensor]:
        """
        仅使用第一个Q网络计算Q值，用于策略更新。

        :param obs: 全局观察张量
        :param actions: 全局动作张量
        :return: 每个智能体的第一个Q网络的Q值列表
        """
        q_value_list = []
        features = []

        # 提取特征
        for agent_id in range(self.n_agents):
            with th.set_grad_enabled(not self.share_features_extractor):
                agent_obs = self._agent_obs_tensor_extract(agent_id, obs)
                agent_features = self.extract_features(agent_id, agent_obs, self.features_extractor_list[agent_id])
            features.append(agent_features)

        # 合并特征和动作
        features = th.cat(features, dim=-1)
        qvalue_input = th.cat([features, actions], dim=1)

        # 仅使用第一个Q网络
        for agent_id in range(self.n_agents):
            q_value_list.append(self.q_networks_list[agent_id][0](qvalue_input))

        return q_value_list


class MADDPGPolicy(MultiAgentBasePolicy):
    """
    多智能体深度确定性策略梯度 (MADDPG) 策略类

    MADDPG算法的特点是:
    1. 每个智能体基于自己的局部观察选择动作(去中心化执行)
    2. Critic评估基于全局状态和所有智能体的联合动作(中心化训练)
    3. 使用目标网络进行稳定学习
    4. 支持每个智能体拥有独特的网络架构和特征提取器

    :param n_agents: 智能体数量
    :param observation_space: 所有智能体的组合观察空间
    :param action_space: 所有智能体的组合动作空间
    :param observation_space_list: 每个智能体的观察空间列表
    :param action_space_list: 每个智能体的动作空间列表
    :param lr_schedule_list: 每个智能体的学习率调度器列表
    :param net_arch: 每个智能体的网络架构列表，可以是层列表或包含actor/critic键的字典列表
    :param activation_fn: 激活函数，默认为ReLU
    :param features_extractor_class_list: 每个智能体的特征提取器类列表
    :param features_extractor_kwargs_list: 每个智能体的特征提取器参数列表
    :param normalize_images: 是否归一化图像输入，默认为True
    :param optimizer_class_list: 每个智能体的优化器类列表
    :param optimizer_kwargs_list: 每个智能体的优化器参数列表
    :param n_critics: 每个智能体的Critic网络数量，用于双Q学习，默认为2
    :param share_features_extractor: 是否在Actor和Critic之间共享特征提取器，默认为False
    """
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        n_agents: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        observation_space_list: list[spaces.Space],
        action_space_list: list[spaces.Space],
        lr_schedule_list: list[Schedule],
        # 每个智能体的网络架构：可以是每个智能体的层列表，或者每个智能体的actor/critic分离架构
        net_arch: Optional[Union[list[list[int]], list[dict[str, list[int]]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class_list: list[type[BaseFeaturesExtractor]] = None,
        features_extractor_kwargs_list: Optional[list[dict[str, Any]]] = None,
        normalize_images: bool = True,
        optimizer_class_list: list[type[th.optim.Optimizer]] = None,
        optimizer_kwargs_list: Optional[list[dict[str, Any]]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            n_agents=n_agents,
            observation_space=observation_space,
            action_space=action_space,
            observation_space_list=observation_space_list,
            action_space_list=action_space_list,
            features_extractor_class_list=features_extractor_class_list,
            features_extractor_kwargs_list=features_extractor_kwargs_list,
            normalize_images=normalize_images,
            optimizer_class_list=optimizer_class_list,
            optimizer_kwargs_list=optimizer_kwargs_list,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            net_arch = [[] for _ in range(n_agents)]
            if features_extractor_class_list is None:
                features_extractor_class_list = [FlattenExtractor for _ in range(n_agents)]
            for agent_id in range(self.n_agents):
                if features_extractor_class_list[agent_id] == NatureCNN:
                    net_arch[agent_id] = [256, 256]
                else:
                    net_arch[agent_id] = [400, 300]

        actor_arch, critic_arch = get_multi_agent_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "n_agents": self.n_agents,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "observation_space_list": self.observation_space_list,
            "action_space_list": self.action_space_list,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule_list)

    def _build(self, lr_schedule_list: list[Schedule]) -> None:
        """
        构建MADDPG策略的网络和优化器

        :param lr_schedule_list: 每个智能体的学习率调度器列表
        """
        # Create actor and target
        self.actor = self.make_actor(features_extractor_list=None)
        self.actor_target = self.make_actor(features_extractor_list=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 初始化优化器
        optimizing_params = []
        for agent_id in range(self.n_agents):
            agent_params = list(self.actor.mu_list[agent_id].parameters()) \
                           + list(self.actor.features_extractor_list[agent_id].parameters())
            optimizing_params.append(agent_params)

        self.actor.init_optimizers(optimizing_params, lr_schedule_list)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor_list=self.actor.features_extractor_list)

            self.critic_target = self.make_critic(features_extractor_list=self.actor.features_extractor_list)
        else:
            self.critic = self.make_critic(features_extractor_list=None)
            self.critic_target = self.make_critic(features_extractor_list=None)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # 初始化优化器
        optimizing_params = []
        for agent_id in range(self.n_agents):
            # 收集当前智能体所有Q网络的参数
            agent_specific_params = []
            for i in range(self.critic.n_critics):
                agent_specific_params.extend(list(self.critic.q_networks_list[agent_id][i].parameters()))
            optimizing_params.append(agent_specific_params)
        self.critic.init_optimizers(optimizing_params, lr_schedule_list)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """获取构造函数参数，用于模型保存和加载"""
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_agents=self.n_agents,
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class_list=self.optimizer_class_list,
                optimizer_kwargs_list=self.optimizer_kwargs_list,
                features_extractor_class_list=self.features_extractor_class_list,
                features_extractor_kwargs_list=self.features_extractor_kwargs_list,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor_list: Optional[list[BaseFeaturesExtractor]] = None) -> Actor:
        """
        创建Actor网络

        :param features_extractor_list: 可选的特征提取器列表，如果为None则创建新的
        :return: 创建的Actor网络
        """
        actor_kwargs = self._update_features_extractors(self.actor_kwargs, features_extractor_list)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor_list: Optional[list[BaseFeaturesExtractor]] = None) -> ContinuousCritic:
        """
        创建Critic网络

        :param features_extractor_list: 可选的特征提取器列表，如果为None则创建新的
        :return: 创建的Critic网络
        """
        critic_kwargs = self._update_features_extractors(self.critic_kwargs, features_extractor_list)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        前向传播，返回所有智能体的联合动作

        :param observation: 全局观察
        :param deterministic: 是否使用确定性策略
        :return: 所有智能体的联合动作
        """
        return self._predict(observation, deterministic=deterministic)

    def _agent_predict(self, agent_id: int, agent_observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        预测单个智能体的动作

        :param agent_id: 智能体ID
        :param agent_observation: 智能体的观察
        :param deterministic: 是否使用确定性策略(在MADDPG中总是确定性的)
        :return: 智能体的动作
        """
        # 注意：MADDPG中deterministic参数被忽略，预测总是确定性的
        return self.actor._agent_predict(agent_id, agent_observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = MADDPGPolicy



