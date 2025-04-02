"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Any, ClassVar, Optional, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from core.common import utils
from core.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from core.common.env_util import is_wrapped
from core.common.logger import Logger
from core.common.monitor import Monitor
from core.common.noise import ActionNoise
from core.common.policies import BasePolicy
from core.common.multi_agent_policies import MultiAgentBasePolicy
from core.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from core.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from core.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from core.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from core.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from core.common.vec_env.patch_gym import _convert_space, _patch_env

SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")
SelfMultiAgentBaseAlgorithm = TypeVar("SelfMultiAgentBaseAlgorithm", bound="MultiAgentBaseAlgorithm")


def maybe_make_env(env: Union[GymEnv, str], verbose: int) -> GymEnv:
    """If env is a string, make the environment; otherwise, return env.

    :param env: The environment to learn from.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating if environment is created
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        env_id = env
        if verbose >= 1:
            print(f"Creating environment from the given name '{env_id}'")
        # Set render_mode to `rgb_array` as default, so we can record video
        try:
            env = gym.make(env_id, render_mode="rgb_array")
        except TypeError:
            env = gym.make(env_id)
    return env


class BaseAlgorithm(ABC):
    """
    The base of RL algorithms

    :param policy (策略函数): The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env (学习的环境): The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate (学习率，可以是训练进度的函数): learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param policy_kwargs (策略函数的超参数): Additional arguments to be passed to the policy on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    # Policy aliases (see _get_policy_from_name())
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {}
    policy: BasePolicy
    observation_space: spaces.Space
    action_space: spaces.Space
    n_envs: int
    lr_schedule: Schedule
    _logger: Logger

    def __init__(
        self,
        policy: Union[str, type[BasePolicy]],
        env: Union[GymEnv, str, None],
        learning_rate: Union[float, Schedule],
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
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ) -> None:
        """
        在这个初始化函数中重点是对环境进行了预处理，也就是说如果环境在外部没有被向量化和标准化等，
        在这里都会统一进行包装
        """
        # 初始化策略类型，可以用别名获取也可以直接传入
        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # 现在是第几步
        self.num_timesteps = 0

        # Used for updating schedules
        self._total_timesteps = 0

        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise: Optional[ActionNoise] = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log

        # 上一个观测值
        self._last_obs = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]

        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._episode_num = 0

        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]

        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int

        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env: Optional[VecEnv] = None
        self._vec_normalize_env: Optional[VecNormalize] = None

        # Create and wrap the env if needed
        if env is not None:
            env = maybe_make_env(env, self.verbose)
            # 将环境进行向量化包装，也就是说无论你之前有没有进行向量化包装环境，在这里都会默认的包装一次
            env = self._wrap_env(env, self.verbose, monitor_wrapper)
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.n_envs = env.num_envs
            self.env = env

            # get VecNormalize object if needed
            # 如果没有被归一化包装的话在这里会进行默认的归一化包装
            self._vec_normalize_env = unwrap_vec_normalize(env)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, spaces.Dict):
                raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")

            if self.use_sde and not isinstance(self.action_space, spaces.Box):
                raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space.low, self.action_space.high]))
                ), "Continuous action space must have a finite lower and upper bound"

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            # 将gym转换成gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        # 确保环境的字典观察空间没有嵌套，这个是不支持的
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger: Logger) -> None:
        """
        将日志记录器换成自定义的日志记录器
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[list[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _excluded_save_params(self) -> list[str]:
        """
        这个函数 返回一个列表，列出在序列化（Pickle 保存）时应该被排除的参数。
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",  # 策略网络，通常使用 `torch.save()`
            "device",  # 设备（CPU/GPU），Pickle 可能无法正确保存
            "env",  # 训练环境对象，可能无法序列化
            "replay_buffer",  # 经验重放缓冲区，数据量大，不适合 Pickle
            "rollout_buffer",  # 轨迹缓冲区
            "_vec_normalize_env",  # 归一化环境对象
            "_episode_storage",  # 训练时存储的 episode 数据
            "_logger",  # 记录日志的对象
            "_custom_logger",  # 自定义日志对象
        ]

    def _get_policy_from_name(self, policy_name: str) -> type[BasePolicy]:
        """
        Get a policy class from its name representation.

        The goal here is to standardize policy naming, e.g.
        all algorithms can call upon "MlpPolicy" or "CnnPolicy",
        and they receive respective policies that work for them.

        :param policy_name: Alias of the policy
        :return: A policy class (type)
        """

        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        该函数 获取 PyTorch 变量的名称，以便用 torch.save() 进行保存，而不是使用 Pickle 进行序列化。
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy"]

        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        # 只有当 reset_num_timesteps=True 或者 self._last_obs 为空时，才会重新重置环境（调用 env.reset()）。
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: list[dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
        self: SelfBaseAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBaseAlgorithm:
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: for on-policy algos (e.g., PPO, A2C, ...) this is the number of
            training iterations (i.e., log_interval * n_steps * n_envs timesteps) before logging;
            for off-policy algos (e.g., TD3, SAC, ...) this is the number of episodes before
            logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(  # noqa: C901
        cls: type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            n_params_saved = len(saved_optim_params)
            n_params = len(model.policy.optimizer.param_groups[0]["params"])
            if n_params_saved == 2 * n_params:
                # Truncate to include only online network params
                params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[:n_params]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def get_parameters(self) -> dict[str, dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)
        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


class MultiAgentBaseAlgorithm(ABC):
    """
    多智能体强化学习基类，所有具体算法都将继承自该类，
    该类定义了通用的算法结构，包括策略选择、环境管理、学习率调度等。
    ---------------------------------------------------------------------------------
    """
    # 定义策略别名，用于映射字符串名称到具体策略类 [see _get_policy_from_name()]
    policy_aliases: ClassVar[dict[str, type[MultiAgentBasePolicy]]] = {}
    policy: MultiAgentBasePolicy  # 具体的策略模型
    observation_space: spaces.Space  # 观测空间
    action_space: spaces.Space  # 动作空间
    n_envs: int  # 环境数量（多环境并行训练）
    lr_schedule_list: list[Schedule]  # 学习率调度器
    _logger: Logger  # 记录日志

    def __init__(
        self,
        n_agents: int,                          # 智能体个数
        policy: Union[str, type[MultiAgentBasePolicy]],   # 策略模型，可以是字符串或类的列表
        env: Union[GymEnv, str, None],          # 交互环境，可以是字符串（Gym ID）或环境实例
        learning_rates: Union[list[float], list[Schedule]],  # 学习率，可固定或动态调整
        policy_kwargs: Optional[dict[str, Any]] = None,     # 传递给策略的额外参数
        stats_window_size: int = 100,                       # 统计窗口大小（计算平均奖励等）
        tensorboard_log: Optional[str] = None,              # TensorBoard 日志存储位置
        verbose: int = 0,                                   # 日志输出等级（0=不输出，1=基本信息，2=详细信息）
        device: Union[th.device, str] = "auto",             # 计算设备（CPU/GPU 自动选择）
        support_multi_env: bool = False,                    # 是否支持多环境并行训练
        monitor_wrapper: bool = True,                       # 是否使用 Monitor 包装环境
        seed: Optional[int] = None,                         # 随机种子（保证实验可复现）
        use_sde: bool = False,                              # 是否使用状态依赖探索（gSDE）
        sde_sample_freq: int = -1,                          # gSDE 采样频率（默认 -1 仅在 rollout 开始时采样）
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,   # 该算法支持的动作空间类型
    ):
        self.n_agents = n_agents
        # 处理策略：如果 `policy` 是字符串，则从别名字典获取相应的策略类
        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        if not (n_agents == len(learning_rates)):
            raise ValueError(f"In a multi-agent scenario, the number of [n_agents, len(learning_rates)] "
                             f" must be consistent, now they are [{n_agents}, "
                             f"{len(learning_rates)}], respectively.")

        self.learning_rate_list = learning_rates
        # 策略网络的参数存储
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # 自动选择计算设备（CPU/GPU）
        self.device = get_device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        # 日志等级
        self.verbose = verbose

        self.num_timesteps = 0  # 训练过程中已经完成的步数
        self._total_timesteps = 0  # 训练的目标步数
        self._num_timesteps_at_start = 0  # 训练起始步数（用于计算 FPS）, it is updated at each call of learn()
        self.seed = seed  # 随机种子，确保实验可复现
        self.action_noise: Optional[ActionNoise] = None
        self.start_time = 0.0  # 训练开始的时间戳
        self.tensorboard_log = tensorboard_log  # TensorBoard 日志存储路径

        # 记录上一次观测的状态变量
        self._last_obs = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        # 上一次 episode 是否开始（用于多环境）
        self._last_episode_starts = None  # type: Optional[np.ndarray]

        # When using VecNormalize:
        # 如果使用 VecNormalize 归一化环境，需要记录原始观测数据
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]

        # 记录当前 episode 的编号（用于统计）
        self._episode_num = 0

        # Used for gSDE only
        self.use_sde = use_sde  # 是否使用状态依赖探索（gSDE），它指的是噪声的采样依赖于当前的状态，而不是使用固定的随机噪声
        self.sde_sample_freq = sde_sample_freq  # gSDE 采样频率
        self._current_progress_remaining = 1.0  # 训练进度（从 1 递减到 0），这用来更新学习率

        # Buffers for logging
        self._stats_window_size = stats_window_size  # 统计窗口大小（计算奖励等）
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        self._n_updates = 0  # type: int        # 记录模型参数更新的次数（适用于 TD3 等，模型延迟更新）

        # Whether the user passed a custom logger or not
        self._custom_logger = False  # 个性化的日志记录器
        self.env: Optional[VecEnv] = None  # 训练环境，可能为空（在 `learn()` 中设置）
        self._vec_normalize_env: Optional[VecNormalize] = None  # 归一化环境（如 `VecNormalize`）

        # Create and wrap the env if needed
        # 如果传入了环境，进行初始化
        if env is not None:
            env = maybe_make_env(env, self.verbose)  # 创建 Gym 环境（如果是字符串 ID）
            env = self._wrap_env(env, self.verbose, monitor_wrapper)  # 进一步包装环境，使其可以支持多环境并行训练，monitor记录，图像通道调整等
            self.observation_space = env.observation_space  # 状态空间
            self.action_space = env.action_space  # 动作空间
            self.n_envs = env.num_envs  # 环境个数（支持多环境）
            self.env = env

            # get VecNormalize object if needed
            # 这行代码的作用是检查并提取 VecNormalize 归一化封装的环境，以便 BaseAlgorithm 访问未归一化的原始环境数据。
            self._vec_normalize_env = unwrap_vec_normalize(env)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy in ["MultiAgentMlpPolicy", "MultiAgentCnnPolicy"] and isinstance(self.observation_space, spaces.Dict):
                raise ValueError(
                    f"You must use `MultiAgentMultiInputPolicy` when working with dict observation space, not {policy}")

            if self.use_sde and not isinstance(self.action_space, spaces.Box):
                raise ValueError(
                    "generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space.low, self.action_space.high]))
                ), "Continuous action space must have a finite lower and upper bound"

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.
        --------------------------------------------------------------------------------------
        VecEnv（Vectorized Environment）是一种用于并行运行多个环境的封装。
        它允许强化学习算法在多个环境实例上同时执行交互，以提高采样效率并减少训练时间。

        :param env:
        :param verbose: 控制日志输出级别（0：无输出，1：输出封装信息）
        :param monitor_wrapper: 是否使用 Monitor 进行封装，以记录回合奖励等信息。
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type:

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                            is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule_list = []
        for learning_rate in self.learning_rate_list:
            self.lr_schedule_list.append(get_schedule_fn(learning_rate))

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: list[Union[th.optim.Optimizer, list[th.optim.Optimizer]]]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        for i, lr_schedule in enumerate(self.lr_schedule_list):
            # Log the current learning rate
            self.logger.record(f"train/agent_{i}_learning_rate", lr_schedule(self._current_progress_remaining))

            if not isinstance(optimizers, list):
                raise ValueError("Expected a list of optimizers for multi agents")

            optimizer_item = optimizers[i] if i < len(optimizers) else None
            if optimizer_item is None:
                continue

            if isinstance(optimizer_item, list):
                for optimizer in optimizer_item:
                    update_learning_rate(optimizer, lr_schedule(self._current_progress_remaining))
            elif isinstance(optimizer_item, th.optim.Optimizer):
                update_learning_rate(optimizer_item, lr_schedule(self._current_progress_remaining))
            else:
                raise ValueError(f"Optimizer update fails, expecting all optimizers for each agent to form a list")

    def _excluded_save_params(self) -> list[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_policy_from_name(self, policy_name: str) -> type[BasePolicy]:
        """
        根据策略名称获取对应的策略类。
        这个方法的目的是提供一个统一的策略命名方式，使所有强化学习算法可以使用
        诸如 "MlpPolicy" 或 "CnnPolicy" 这样的通用字符串来获取相应的策略类。
        :param policy_name: 策略的别名（如 "MlpPolicy" 或 "CnnPolicy"）
        :return: 返回一个策略类（而不是类的实例）
        """

        # 判断提供的 policy_name 是否在 policy_aliases 字典中
        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]             # 返回对应的策略类
        else:
            raise ValueError(f"Policy {policy_name} unknown")   # 如果 policy_name 不在已注册的策略别名字典中，则抛出异常

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy"]

        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: list[dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    @abstractmethod
    def learn(
            self: SelfMultiAgentBaseAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfMultiAgentBaseAlgorithm:
        """
        执行完整的训练循环，控制数据收集、策略更新和日志记录

        :param total_timesteps: 总训练步数
        :param callback: callback(s) called at every step with state of the algorithm.，回调函数，用于自定义训练逻辑
        :param log_interval: for on-policy algos (e.g., PPO, A2C, ...) this is the number of
            training iterations (i.e., log_interval * n_steps * n_envs timesteps) before logging;
            for off-policy algos (e.g., TD3, SAC, ...) this is the number of episodes before
            logging.日志记录的间隔
        :param tb_log_name: the name of the run for TensorBoard logging。TensorBoard 日志名称。
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        根据当前策略选择动作

        :param observation: the input observation观测值
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # 若是列表保存的就是优化器序列
                if isinstance(attr, list):
                    # 重新载入每个优化器的状态
                    for i, opt in enumerate(attr):
                        opt.load_state_dict(params[name][f'optimizer_{i}'])
                else:
                    # Assume attr is th.nn.Module
                    attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(  # noqa: C901
            cls: type[SelfMultiAgentBaseAlgorithm],
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfMultiAgentBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]
            else:
                raise ValueError(f"missing 'env' in data")

        model = cls(
            n_agents=data["n_agents"],
            observation_splits=data["observation_splits"],
            action_splits=data["action_splits"],
            learning_rate_list=data["learning_rate_list"],
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            n_params_saved = len(saved_optim_params)
            n_params = len(model.policy.optimizer.param_groups[0]["params"])
            if n_params_saved == 2 * n_params:
                # Truncate to include only online network params
                params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[
                                                                          :n_params]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def get_parameters(self) -> dict[str, dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        # print(self.__dict__)
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            if isinstance(attr, list):
                optimizer_states = {f'optimizer_{i}': opt.state_dict() for i, opt in enumerate(attr)}
                params[name] = optimizer_states
            else:
                params[name] = attr.state_dict()
        return params

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved。为保存路径（文件或文件夹）
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


