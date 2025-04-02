from typing import Any, Optional, Union, Tuple

import warnings
import torch as th
from gymnasium import spaces
from torch import nn

from core.common.policies import BasePolicy, ContinuousCritic
from core.common.preprocessing import get_action_dim
from core.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from core.common.type_aliases import PyTorchObs, Schedule


class BehaviorVAE(nn.Module):
    """
    Variational Auto-Encoder for BCQ to model the behavior policy.

    :param state_dim: Dimension of the state space
    :param action_dim: Dimension of the action space
    :param latent_dim: Dimension of the latent space
    :param hidden_dim: Dimension of the hidden layers
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            latent_dim: int = None,
            hidden_dim: int = 750
    ):
        super(BehaviorVAE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        if latent_dim is None:
            latent_dim = 2 * action_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Assuming actions are in [-1, 1]
        )

    def forward(self, state: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Encode state-action pair and decode to reconstruct action.

        :param state: State tensor
        :param action: Action tensor
        :return: Reconstructed action, mean and std of the latent distribution
        """
        # Encode
        z = self.encoder(th.cat([state, action], dim=1))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = log_std.exp()

        # Sample from latent space
        z = mean + std * th.randn_like(std)

        # Decode
        reconstructed_action = self.decoder(th.cat([state, z], dim=1))

        return reconstructed_action, mean, std

    def encode(self, state: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Encode state-action pair to latent distribution parameters.

        :param state: State tensor
        :param action: Action tensor
        :return: Mean and std of the latent distribution
        """
        z = self.encoder(th.cat([state, action], dim=1))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        return mean, log_std.exp()

    def decode(self, state: th.Tensor, z: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Decode latent vector to action.

        :param state: State tensor
        :param z: Latent vector (if None, sample from standard normal)
        :return: Reconstructed action
        """
        if z is None:
            z = th.randn((state.shape[0], self.latent_dim)).to(state.device).clamp(-0.5, 0.5)
        return self.decoder(th.cat([state, z], dim=1))

    def sample_action(self, state: th.Tensor, num_samples: int = 10) -> th.Tensor:
        """
        Sample actions from the VAE for a given state.

        :param state: State tensor
        :param num_samples: Number of actions to sample
        :return: Sampled actions
        """
        state_rep = state.repeat(num_samples, 1)
        z = th.randn((state_rep.shape[0], self.latent_dim)).to(state.device).clamp(-0.5, 0.5)
        return self.decoder(th.cat([state_rep, z], dim=1))


class PerturbationNetwork(nn.Module):
    """
    Perturbation network for BCQ to add small perturbations to the actions.

    :param state_dim: Dimension of the state space
    :param action_dim: Dimension of the action space
    :param hidden_dim: Dimension of the hidden layers
    :param max_perturbation: Maximum perturbation magnitude
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dim: int,
            max_perturbation: float = 0.05
    ):
        super(PerturbationNetwork, self).__init__()

        self.max_perturbation = max_perturbation

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output is scaled later
        )

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Add perturbation to the action.

        :param state: State tensor
        :param action: Action tensor
        :return: Perturbed action
        """
        perturbation = self.model(th.cat([state, action], dim=1)) * self.max_perturbation
        return (action + perturbation).clamp(-1, 1)  # Assuming actions are in [-1, 1]


class VAEActor(BasePolicy):
    """
    Actor network (policy) for BCQ using VAE.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: Optional[Union[list[int], dict[str, list[Union[int, float]]]]],
        features_extractor: nn.Module,
        features_dim: int,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.features_dim = features_dim

        action_dim = get_action_dim(self.action_space)

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(vae_latent_dim=..., vae_hidden_dim=..., perturbation_hidden_dim=..., "
                    "max_perturbation=...) instead of net_arch=[dict(...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = dict(vae_latent_dim=32, vae_hidden_dim=720,
                            perturbation_hidden_dim=400, max_perturbation=0.05)

        self.net_arch = net_arch

        self.vae = BehaviorVAE(state_dim=self.features_dim,
                               action_dim=action_dim,
                               latent_dim=net_arch["vae_latent_dim"],
                               hidden_dim=net_arch["vae_hidden_dim"])

        # Perturbation network to add small changes to VAE actions
        self.perturbation = PerturbationNetwork(state_dim=self.features_dim,
                                                action_dim=action_dim,
                                                hidden_dim=net_arch["perturbation_hidden_dim"],
                                                max_perturbation=net_arch["max_perturbation"])

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, num_samples: int = 10) -> th.Tensor:
        # assert deterministic, 'The BCQ actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        features_rep = features.repeat(num_samples, 1)

        action_candidates = self.vae.sample_action(features, num_samples)
        # Perturbation
        perturbed_actions = self.perturbation(features_rep, action_candidates)

        return perturbed_actions

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of BCQ.
        # Predictions are always deterministic.
        return self(obs=observation, num_samples=100)


class BCQPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for BCQ.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: VAEActor
    actor_target: VAEActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        actor_net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        critic_net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture
        if actor_net_arch is None:
            actor_net_arch = dict(vae_latent_dim=32, vae_hidden_dim=64,
                                  perturbation_hidden_dim=64, max_perturbation=0.05)
        else:
            assert isinstance(actor_net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
            assert "vae_latent_dim" in actor_net_arch, "Error: no key 'vae_latent_dim' was provided in net_arch for the actor network"
            assert "vae_hidden_dim" in actor_net_arch, "Error: no key 'vae_hidden_dim' was provided in net_arch for the actor network"
            assert "perturbation_hidden_dim" in actor_net_arch, "Error: no key 'perturbation_hidden_dim' was provided in net_arch for the actor network"
            assert "max_perturbation" in actor_net_arch, "Error: no key 'max_perturbation' was provided in net_arch for the actor network"

        if critic_net_arch is None:
            if features_extractor_class == NatureCNN:
                critic_net_arch = [256, 256]
            else:
                critic_net_arch = [400, 300]

        self.actor_arch = actor_net_arch
        self.critic_arch = critic_net_arch

        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_net_arch,
            "normalize_images": normalize_images,
        }

        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()

        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_net_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.perturbation_optimizer = self.optimizer_class(
            self.actor.perturbation.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        self.actor.vae_optimizer = self.optimizer_class(
            [p for p in self.actor.parameters() if id(p) not in {id(p1) for p1 in list(self.actor.perturbation.parameters())}],
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
                observation_space=self.observation_space,
                action_space=self.action_space,
                actor_net_arch=self.actor_arch,
                activation_fn=self.activation_fn,
                normalize_images=self.normalize_images,
                critic_arch=self.critic_arch,
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        return data

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of BCQ.
        #   Predictions are always deterministic.
        num_samples = 100
        candidate_actions = self.actor(observation, num_samples=num_samples)
        observations_rep = observation.repeat(num_samples, 1)
        q1 = self.critic.q1_forward(observations_rep, candidate_actions)
        ind = q1.argmax(0)

        return candidate_actions[ind]

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> VAEActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return VAEActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = BCQPolicy



