🚀 Extended Stable-Baselines3 for Offline and Multi-Agent Reinforcement Learning

This repository extends Stable-Baselines3 by incorporating additional reinforcement learning algorithms, including:

🧠 BCQ (Batch-Constrained Q-Learning): An offline reinforcement learning algorithm that mitigates the issue of overestimation in value-based methods when learning from offline datasets.

🤖 IDDPG (Independent Deep Deterministic Policy Gradient): A multi-agent reinforcement learning algorithm where agents independently learn policies using DDPG.

🤝 MADDPG (Multi-Agent Deep Deterministic Policy Gradient): A centralized training and decentralized execution framework for cooperative multi-agent reinforcement learning.

✨ Features

✅ Consistent API: Maintains the same usage style as Stable-Baselines3 for seamless integration.

📂 Prebuilt Experiment Cases: Provides example usage in the experiments directory to help users quickly get started.

🔄 Ongoing Updates: Continual improvements and additions of new reinforcement learning algorithms.

📥 Installation

To install the extended library:

pip install git+https://github.com/CHAINNEVERLIU/Pytorch-RL-EnhancedStableBaselines.git

🛠 Usage

The extended algorithms can be used just like other Stable-Baselines3 models:

from core import BCQ, IDDPG, MADDPG

model = BCQ("MlpPolicy", env)
model.learn(total_timesteps=100000)

For detailed examples, check the experiments directory.

🤝 Contributions & Future Work

This project is actively maintained, and new algorithms will be added in future updates. Contributions are welcome! 🚀

For any questions or discussions, feel free to open an issue or submit a pull request. 📝


