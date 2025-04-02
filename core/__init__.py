import os

from core.a2c import A2C
from core.common.utils import get_system_info
from core.ddpg import DDPG
from core.dqn import DQN
from core.her.her_replay_buffer import HerReplayBuffer
from core.ppo import PPO
from core.sac import SAC
from core.td3 import TD3
from core.bcq import BCQ
from core.iddpg import IDDPG
from core.maddpg import MADDPG

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )


__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "HerReplayBuffer",
    "get_system_info",
    "BCQ",
    "IDDPG",
    "MADDPG",
]
