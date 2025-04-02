from typing import Any, ClassVar, Optional, TypeVar, Union, Dict

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from torch.nn import functional as F

from core.common.buffers import ReplayBuffer
from core.common.noise import ActionNoise
from core.common.offline_policy_algorithm import OfflineAlgorithm
from core.common.policies import BasePolicy, ContinuousCritic
from core.common.type_aliases import GymEnv, MaybeCallback, Schedule
from core.common.utils import get_parameters_by_name, polyak_update
from core.bcq.policies import MlpPolicy, BCQPolicy, VAEActor


SelfBCQ = TypeVar("SelfBCQ", bound="BCQ")


class BCQ(OfflineAlgorithm):
    """
    Batch-Constrained deep Q-learning (BCQ) implementation.

    Paper: https://arxiv.org/abs/1812.02900

    :param policy: Policy class
    :param env: The environment to evaluate on
    :param dataset: Path to the offline dataset or the dataset object itself
    :param learning_rate: learning rate for the optimizer
    :param buffer_size: size of the replay buffer for dataset loading
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param gradient_steps: How many gradient steps to do after each evaluation step
    :param behavior_cloning_warmup: Number of gradient steps to perform behavior cloning before RL training
    :param n_eval_episodes: Number of episodes to evaluate the policy on
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param stats_window_size: Window size for the evaluation logging
    :param tensorboard_log: the log location for tensorboard
    :param verbose: Verbosity level
    :param device: Device on which the code should run
    :param seed: Seed for the pseudo random generators
    """
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
    }
    policy: BCQPolicy
    actor: VAEActor
    actor_target: VAEActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
            self,
            policy: Union[str, type[BasePolicy]],
            env: Union[GymEnv, str, None],
            dataset: Union[str, ReplayBuffer] = None,
            learning_rate: Union[float, Any] = 3e-4,
            buffer_size: int = 1_000_000,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            gradient_steps: int = 1,
            behavior_cloning_warmup: int = 0,
            n_eval_episodes: int = 10,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            verbose: int = 0,
            device: Union[th.device, str] = "auto",
            seed: Optional[int] = None,
            actor_delay: int = 2,
            _init_setup_model: bool = True,
    ):
        # Initialize the parent class
        super(BCQ, self).__init__(
            policy=policy,
            env=env,
            dataset=dataset,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            gradient_steps=gradient_steps,
            dataset_buffer_class=None,
            dataset_buffer_kwargs=None,
            n_eval_episodes=n_eval_episodes,
            behavior_cloning_warmup=behavior_cloning_warmup,
            conservative_weight=0.0,  # BCQ doesn't use conservative weight
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
        )

        self.actor_delay = actor_delay

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Create networks, buffer and optimizers.
        Override to initialize BCQ-specific components.
        """
        super()._setup_model()

        # Extract dimensions from observation and action spaces
        if isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("BCQ currently does not support Dict observation spaces")

        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.perturbation_optimizer, self.actor.vae_optimizer, self.critic.optimizer])

        actor_losses, vae_losses, critic_losses = [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # VAE forward pass
            reconstructed_actions, mean, std = self.actor.vae(state=replay_data.observations, action=replay_data.actions)
            # Compute reconstruction loss
            recon_loss = F.mse_loss(reconstructed_actions, replay_data.actions)
            # Compute KL divergence
            kl_loss = -0.5 * (1 + th.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            # Total loss
            vae_loss = recon_loss + 0.5 * kl_loss

            # Optimize
            self.actor.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.actor.vae_optimizer.step()

            vae_losses.append(vae_loss.item())

            with th.no_grad():
                # VAE 网络不需要使用目标网络，所以每次更新之后它的参数需要覆盖
                self.actor_target.vae.load_state_dict(self.actor.vae.state_dict())
                training_num_samples = 10
                next_candidate_actions = self.actor_target(replay_data.next_observations, num_samples=training_num_samples)

                next_observations_rep = replay_data.next_observations.repeat(training_num_samples, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(next_observations_rep, next_candidate_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # Reshape and take maximum Q-value
                next_q_values = next_q_values.reshape(batch_size, training_num_samples)
                next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.actor_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                     self.actor(replay_data.observations, num_samples=1)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.perturbation_optimizer.zero_grad()
                actor_loss.backward()
                self.actor.perturbation_optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # VAE 网络不需要使用目标网络，所以每次更新之后它的参数需要覆盖
                self.actor_target.vae.load_state_dict(self.actor.vae.state_dict())

                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if len(actor_losses) > 0:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            self.logger.record("train/vae_loss", np.mean(vae_losses))

    def learn(
        self: SelfBCQ,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "BCQ",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> OfflineAlgorithm:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.vae_optimizer", "actor.perturbation_optimizer", "critic.optimizer"]
        return state_dicts, []

    def _behavior_cloning_update(self, observations: np.ndarray, actions: np.ndarray) -> float:
        pass

    def _behavior_cloning_warmup(self, callback) -> None:
        pass

