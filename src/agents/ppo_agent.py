"""PPO agent wrapper around Stable-Baselines3."""

from pathlib import Path

from stable_baselines3 import PPO


class PPOAgent:
    """Wraps SB3 PPO for training and inference."""

    def __init__(self, env, config: dict | None = None):
        cfg = config or {}
        train_cfg = cfg.get("training", {})

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            n_steps=train_cfg.get("n_steps", 2048),
            batch_size=train_cfg.get("batch_size", 64),
            n_epochs=train_cfg.get("n_epochs", 10),
            gamma=train_cfg.get("gamma", 0.99),
            verbose=1,
            seed=train_cfg.get("seed", 42),
        )

    def train(self, total_timesteps: int = 50000):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def predict(self, obs):
        action, state = self.model.predict(obs, deterministic=True)
        return action, state

    @classmethod
    def load(cls, path: str, env):
        agent = cls.__new__(cls)
        agent.model = PPO.load(path, env=env)
        return agent
