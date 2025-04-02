from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from core.common.buffers import ReplayBuffer
from core.common.noise import ActionNoise
from core.common.multiagent_policy_algorithm import OffMultiAgentPolicyAlgorithm
from core.common.multi_agent_policies import MultiAgentBasePolicy
from core.common.type_aliases import GymEnv, MaybeCallback, Schedule
from core.common.utils import get_parameters_by_name, polyak_update
from core.iddpg.policies import Actor, MlpPolicy, IDDPGPolicy, ContinuousCritic


SelfIDDPG = TypeVar("SelfIDDPG", bound="IDDPG")


class IDDPG(OffMultiAgentPolicyAlgorithm):
    """
    IDDPG算法
    """

    policy_aliases: ClassVar[dict[str, type[MultiAgentBasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
    }
    policy: IDDPGPolicy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        n_agents: int,
        policy: Union[str, type[IDDPGPolicy]],
        env: Union[GymEnv, str],
        observation_splits: list[list[int]],
        action_splits: list[list[int]],
        learning_rate_list: Union[list[float], list[Schedule]] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            n_agents=n_agents,
            policy=policy,
            env=env,
            observation_splits=observation_splits,
            action_splits=action_splits,
            learning_rate_list=learning_rate_list,
            buffer_size=buffer_size,  # 1e6
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            monitor_wrapper=True,
            seed=seed,
            use_sde=False,
            sde_support=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        for agent_id in range(self.n_agents):
            # Update learning rate according to lr schedule
            self._update_learning_rate([self.actor.optimizer_list[agent_id], self.critic.optimizer_list[agent_id]])

        actor_losses = [[] for _ in range(self.n_agents)]
        critic_losses = [[] for _ in range(self.n_agents)]
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # 计算下一个时间步的动作
            next_actions_list = []
            for agent_id in range(self.n_agents):
                agent_next_observations = self.actor._agent_obs_tensor_extract(agent_id, replay_data.next_observations)
                agent_action = self.actor._agent_action_tensor_extract(agent_id, replay_data.actions)
                with th.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = agent_action.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target.mu_list[agent_id](agent_next_observations) + noise).clamp(-1, 1)

                next_actions_list.append(next_actions)

            next_actions = th.cat(next_actions_list, dim=-1)
            for agent_id in range(self.n_agents):
                agent_observations = self.actor._agent_obs_tensor_extract(agent_id, replay_data.observations)
                with th.no_grad():
                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions)[agent_id], dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                current_q_values = self.critic(replay_data.observations, replay_data.actions)[agent_id]

                # Compute critic loss
                critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                assert isinstance(critic_loss, th.Tensor)
                critic_losses[agent_id].append(critic_loss.item())

                # Optimize the critics
                self.critic.optimizer_list[agent_id].zero_grad()
                critic_loss.backward()
                self.critic.optimizer_list[agent_id].step()

                # Delayed policy updates
                if self._n_updates % self.policy_delay == 0:
                    actions = []
                    for id in range(self.n_agents):
                        agent_action = self.actor.mu_list[id](agent_observations)
                        actions.append(agent_action)
                    actions = th.cat(actions, dim=-1)
                    # Compute actor loss
                    actor_loss = -self.critic.q1_forward(replay_data.observations, actions)[agent_id].mean()
                    actor_losses[agent_id].append(actor_loss.item())
                    # Optimize the actor
                    self.actor.optimizer_list[agent_id].zero_grad()
                    actor_loss.backward()
                    self.actor.optimizer_list[agent_id].step()

                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                    # Copy running stats, see GH issue #996
                    polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                    polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for agent_id in range(self.n_agents):
            if len(actor_losses[agent_id]) > 0:
                self.logger.record(f"train/agent_{agent_id}_actor_loss", np.mean(actor_losses[agent_id]))
            self.logger.record(f"train/agent_{agent_id}_critic_loss", np.mean(critic_losses[agent_id]))

    def learn(
        self: SelfIDDPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "IDDPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfIDDPG:
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
        state_dicts = ["policy", "actor.optimizer_list", "critic.optimizer_list"]
        return state_dicts, []




























