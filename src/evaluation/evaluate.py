"""Evaluate all agents on the perishable market environment."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import yaml

from src.agents.ppo_agent import PPOAgent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.environment.market_env import PerishableMarketEnv


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(ROOT_DIR / path) as f:
        return yaml.safe_load(f)


def run_episodes(agent, env, n_episodes: int, seed: int = 123) -> dict:
    """Run agent for n_episodes and collect metrics."""
    rewards = []
    inventories_left = []
    episode_lengths = []

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        rewards.append(total_reward)
        inventories_left.append(info["inventory"])
        episode_lengths.append(steps)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_inventory_left": float(np.mean(inventories_left)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "all_rewards": [float(r) for r in rewards],
    }


def main():
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    n_episodes = eval_cfg.get("n_episodes", 50)
    seed = eval_cfg.get("seed", 123)
    results_dir = ROOT_DIR / eval_cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    env = PerishableMarketEnv(config)

    # Set up agents
    agents = {}
    agents["Random"] = RandomAgent(env)
    agents["RuleBased"] = RuleBasedAgent(env)

    model_path = config.get("training", {}).get(
        "model_save_path", "results/ppo_market_agent"
    )
    model_path = str(ROOT_DIR / model_path)
    if Path(f"{model_path}.zip").exists():
        agents["PPO"] = PPOAgent.load(model_path, env)
    else:
        print(f"Warning: No trained PPO model at {model_path}.zip — skipping PPO.")

    # Evaluate
    all_results = {}
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        results = run_episodes(agent, env, n_episodes, seed)
        all_results[name] = results
        print(
            f"  {name}: mean_reward={results['mean_reward']:.2f} "
            f"(+/- {results['std_reward']:.2f}), "
            f"inventory_left={results['mean_inventory_left']:.1f}"
        )

    # Save results
    output_path = results_dir / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
