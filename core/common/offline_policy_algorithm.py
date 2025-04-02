import io
import pathlib
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Optional, TypeVar, Union, Tuple
import traceback
import numpy as np
import torch as th
from gymnasium import spaces
import gymnasium as gym
from tqdm import tqdm

from core.common.base_class import BaseAlgorithm
from core.common.off_policy_algorithm import OffPolicyAlgorithm
from core.common.buffers import DictReplayBuffer, ReplayBuffer
from core.common.callbacks import BaseCallback
from core.common.noise import ActionNoise, VectorizedActionNoise
from core.common.policies import BasePolicy
from core.common.save_util import load_from_pkl, save_to_pkl
from core.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from core.common.utils import safe_mean, should_collect_more_steps, to_numpy
from core.common.vec_env import VecEnv
from core.her.her_replay_buffer import HerReplayBuffer
from core.common.vec_env import DummyVecEnv, VecNormalize

SelfOfflinePolicyAlgorithm = TypeVar("SelfOfflinePolicyAlgorithm", bound="OfflinePolicyAlgorithm")


# Helper classes for offline RL
class DummyEnv(gym.Env):
    """
    Dummy environment used when no environment is provided but spaces are known.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, {}

    def render(self):
        pass


class OfflineAlgorithm(OffPolicyAlgorithm):
    """
    The base for Offline Reinforcement Learning algorithms (ex: BCQ/CQL/TD3+BC)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to evaluate on (if registered in Gym, can be str.
                Can be None for loading trained models)
    :param dataset: Path to the offline dataset or the dataset object itself
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer for dataset loading
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param gradient_steps: How many gradient steps to do after each evaluation step
    :param dataset_buffer_class: Replay buffer class to use for the offline dataset.
        If ``None``, it will be automatically selected.
    :param dataset_buffer_kwargs: Keyword arguments to pass to the dataset buffer on creation.
    :param n_eval_episodes: Number of episodes to evaluate the policy on
    :param behavior_cloning_warmup: Number of gradient steps to perform behavior cloning before RL training
    :param conservative_weight: Weight for conservative Q-learning loss component (if used)
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param stats_window_size: Window size for the evaluation logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param sde_support: Whether the model support gSDE or not
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
            self,
            policy: Union[str, type[BasePolicy]],
            env: Union[GymEnv, str, None],
            learning_rate: Union[float, Schedule],
            dataset: Union[str, ReplayBuffer] = None,
            buffer_size: int = 1_000_000,  # 1e6
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            gradient_steps: int = 1,
            dataset_buffer_class: Optional[type[ReplayBuffer]] = None,
            dataset_buffer_kwargs: Optional[dict[str, Any]] = None,
            n_eval_episodes: int = 10,
            behavior_cloning_warmup: int = 0,
            conservative_weight: float = 0.0,
            policy_kwargs: Optional[dict[str, Any]] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            verbose: int = 0,
            device: Union[th.device, str] = "auto",
            support_multi_env: bool = False,
            monitor_wrapper: bool = True,
            seed: Optional[int] = None,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            sde_support: bool = False,
            supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        # Set train_freq to None since we don't collect rollouts in offline RL
        # We'll also set a dummy environment if none is provided, since the base class requires one
        dummy_env = None
        if env is None and isinstance(dataset, ReplayBuffer):
            # Create a dummy env from dataset observation/action spaces
            dummy_env = DummyVecEnv([lambda: DummyEnv(
                dataset.observation_space,
                dataset.action_space
            )])
            env = dummy_env

        if isinstance(dataset, ReplayBuffer):
            buffer_size = max(buffer_size, int(dataset.size() * 1.1))
            if self.verbose > 0:
                print(f"Buffer_size is updated to {buffer_size} to accommodate dataset plus margin")

        # Pass None for action_noise as we don't explore in offline RL
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=0,  # We start learning immediately
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=(1, "step"),  # Dummy value, not used in offline RL
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=dataset_buffer_class,
            replay_buffer_kwargs=dataset_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=False,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
        )

        self.dataset = dataset
        self.n_eval_episodes = n_eval_episodes
        self.behavior_cloning_warmup = behavior_cloning_warmup
        self.conservative_weight = conservative_weight
        self._dummy_env = dummy_env

        # Depending on the specific offline RL algorithm, additional parameters can be added
        # These are common parameters found in algorithms like BCQ, CQL, TD3+BC, etc.
        self.bc_loss = None  # Placeholder for behavior cloning loss
        self._n_updates = 0  # Counter for updates, useful for BC warmup phase

    def _setup_model(self) -> None:
        """
        Create networks, buffer and optimizers.
        Override base method to load the dataset into the replay buffer.
        """
        super()._setup_model()

        # If dataset is a string (path), load it into the replay buffer
        if isinstance(self.dataset, str):
            self.load_dataset(self.dataset)
        elif isinstance(self.dataset, ReplayBuffer):
            # If it's already a replay buffer, we copy the data
            self._copy_dataset_to_buffer(self.dataset)
        else:
            raise ValueError(f"Dataset must be a path string or a ReplayBuffer instance, got {type(self.dataset)}")

    def load_dataset(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Load an offline dataset into the replay buffer.

        :param path: Path to the dataset file or directory.
        """
        # Implementation depends on the dataset format
        # Here we assume it's a pickled ReplayBuffer
        try:
            if self.verbose > 0:
                print(f"Loading dataset from {path}")

            # 检查文件是否存在
            if isinstance(path, (str, pathlib.Path)) and not os.path.exists(path):
                raise FileNotFoundError(f"Dataset file not found: {path}")

            temp_buffer = load_from_pkl(path, self.verbose)
            self._copy_dataset_to_buffer(temp_buffer)

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            error_msg = f"Dataset loading failed. Error type: {type(e).__name__}, Message: {str(e)}"
            if self.verbose > 0:
                print(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)

    def _copy_dataset_to_buffer(self, source_buffer: ReplayBuffer) -> None:
        """
        Copy data from source buffer to the replay buffer.

        :param source_buffer: Source replay buffer with offline dataset.
        """
        assert self.replay_buffer is not None, "Replay buffer not initialized"

        # 增加数据验证
        if source_buffer is None or source_buffer.size() == 0:
            raise ValueError("Loaded dataset is empty")

        # Example implementation for standard replay buffer
        # This would need to be adapted for different buffer types
        if isinstance(source_buffer, ReplayBuffer) and isinstance(self.replay_buffer, ReplayBuffer):
            self.replay_buffer = source_buffer
            if self.verbose > 0:
                print(f"Finished copying dataset. Replay buffer contains {self.replay_buffer.size()} transitions")
        else:
            raise ValueError("Incompatible buffer types")

    def learn(
            self: SelfOfflinePolicyAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOfflinePolicyAlgorithm:
        """
        Train the model for the specified number of gradient steps.
        In offline RL, we don't collect new experiences but rather train on the fixed dataset.

        :param total_timesteps: The number of gradient steps to perform
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich
        :return: the trained model
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        # Perform behavior cloning warmup if specified
        if self.behavior_cloning_warmup > 0:
            self._behavior_cloning_warmup(callback)

        # Main training loop
        while self.num_timesteps < total_timesteps:
            # Update policy and critics
            self.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

            self.num_timesteps += self.n_envs

            # 用于在每一个更新步后进行一些处理，在这暂时定义为空处理
            self._on_step()

            # Give access to local variables
            callback.update_locals(locals())

            if not callback.on_step():
                break

            # Log training infos
            if log_interval is not None and self.num_timesteps % log_interval == 0:
                self._dump_logs()

        callback.on_training_end()

        return self

    def _behavior_cloning_warmup(self, callback: BaseCallback) -> None:
        """
        Perform behavior cloning warmup phase.
        This is common in offline RL to initialize the policy.

        :param callback: Callback called at each step
        """
        raise NotImplementedError("Subclasses must implement _behavior_cloning_warmup method")

    def _behavior_cloning_update(self, observations: np.ndarray, actions: np.ndarray) -> float:
        """
        Update policy using behavior cloning.
        This is a simple supervised learning step to match the actions in the dataset.

        :param observations: Observations from the replay buffer
        :param actions: Actions from the replay buffer corresponding to the observations
        :return: Behavior cloning loss
        """
        raise NotImplementedError("Subclasses must implement _behavior_cloning_update method")

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates.
        This method needs to be implemented by the specific algorithm.

        :param gradient_steps: Number of gradient steps to perform
        :param batch_size: Batch size to use for training
        """
        raise NotImplementedError("Each offline RL algorithm must implement its own train method")

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> None:
        """
        Override to prevent rollout collection in offline RL.
        """
        warnings.warn(
            "Offline RL algorithms do not collect rollouts during training. "
            "Use learn() directly or evaluate() to assess the trained policy."
        )

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        # 记录数据集大小和特征
        self.logger.record("dataset/size", self.replay_buffer.size())
        self.logger.record("training/bc_warmup_steps", self.behavior_cloning_warmup)

        # 如果实现了保守学习，可以记录保守权重
        if hasattr(self, 'conservative_weight'):
            self.logger.record("training/conservative_weight", self.conservative_weight)

        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)





