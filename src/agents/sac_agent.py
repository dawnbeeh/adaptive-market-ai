"""SAC agent wrapper around Stable-Baselines3."""

from pathlib import Path

from stable_baselines3 import SAC


class SACAgent:
    """Wraps SB3 SAC for training and inference."""

    def __init__(self, env, config: dict | None = None):
        cfg = config or {}
        train_cfg = cfg.get("training_sac", {})

        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            buffer_size=train_cfg.get("buffer_size", 50000),
            batch_size=train_cfg.get("batch_size", 256),
            tau=train_cfg.get("tau", 0.005),
            gamma=train_cfg.get("gamma", 0.99),
            learning_starts=train_cfg.get("learning_starts", 1000),
            train_freq=train_cfg.get("train_freq", 1),
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
        agent.model = SAC.load(path, env=env)
        return agent
