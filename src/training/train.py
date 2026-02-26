"""Training entry point for PPO agent."""

from pathlib import Path

import yaml

from src.agents.ppo_agent import PPOAgent
from src.environment.market_env import PerishableMarketEnv


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    train_cfg = config.get("training", {})

    print("Creating environment...")
    env = PerishableMarketEnv(config)

    print("Initializing PPO agent...")
    agent = PPOAgent(env, config)

    total_timesteps = train_cfg.get("total_timesteps", 50000)
    print(f"Training for {total_timesteps} timesteps...")
    agent.train(total_timesteps=total_timesteps)

    save_path = train_cfg.get("model_save_path", "results/ppo_market_agent")
    agent.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
