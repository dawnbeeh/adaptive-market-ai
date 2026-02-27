"""Generate visualization plots from evaluation results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
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


def plot_agent_comparison(results: dict, results_dir: Path):
    """Bar chart comparing mean rewards across agents."""
    agents = list(results.keys())
    means = [results[a]["mean_reward"] for a in agents]
    stds = [results[a]["std_reward"] for a in agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, means, yerr=stds, capsize=5, color=["#e74c3c", "#3498db", "#2ecc71"][:len(agents)])
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Agent Performance Comparison")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / "agent_comparison.png", dpi=150)
    plt.close()
    print("Saved agent_comparison.png")


def plot_reward_distribution(results: dict, results_dir: Path):
    """Box plot of reward distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [results[a]["all_rewards"] for a in results]
    ax.boxplot(data, labels=list(results.keys()))
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward Distribution by Agent")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "reward_distribution.png", dpi=150)
    plt.close()
    print("Saved reward_distribution.png")


def plot_episode_trace(env, agents_dict: dict, results_dir: Path, seed: int = 42):
    """Plot a single episode trace showing pricing and inventory behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name, agent in agents_dict.items():
        obs, _ = env.reset(seed=seed)
        inventories = [obs[0]]
        freshnesses = [obs[1]]
        prices_asked = []
        rewards = []

        while True:
            action, _ = agent.predict(obs)
            price_mult = float(np.clip(action[1], 0.5, 2.0))
            prices_asked.append(price_mult * obs[2])

            obs, reward, terminated, truncated, info = env.step(action)
            inventories.append(obs[0])
            freshnesses.append(obs[1])
            rewards.append(reward)

            if terminated or truncated:
                break

        steps = range(len(inventories))
        axes[0, 0].plot(steps, inventories, label=name)
        axes[0, 1].plot(steps, freshnesses, label=name)
        axes[1, 0].plot(range(len(prices_asked)), prices_asked, label=name)
        axes[1, 1].plot(range(len(rewards)), np.cumsum(rewards), label=name)

    axes[0, 0].set_title("Inventory Over Time")
    axes[0, 0].set_ylabel("Units")
    axes[0, 1].set_title("Freshness Over Time")
    axes[0, 1].set_ylabel("Freshness")
    axes[1, 0].set_title("Ask Price Over Time")
    axes[1, 0].set_ylabel("Price ($)")
    axes[1, 1].set_title("Cumulative Reward")
    axes[1, 1].set_ylabel("Reward ($)")

    for ax in axes.flat:
        ax.set_xlabel("Day")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Single Episode Trace", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "episode_trace.png", dpi=150)
    plt.close()
    print("Saved episode_trace.png")


def main():
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    results_dir = ROOT_DIR / eval_cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation results
    results_path = results_dir / "evaluation_results.json"
    if not results_path.exists():
        print("No evaluation results found. Run evaluate.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Generate comparison plots
    plot_agent_comparison(results, results_dir)
    plot_reward_distribution(results, results_dir)

    # Generate episode traces
    env = PerishableMarketEnv(config)
    agents = {}
    agents["Random"] = RandomAgent(env)
    agents["RuleBased"] = RuleBasedAgent(env)

    model_path = config.get("training", {}).get(
        "model_save_path", "results/ppo_market_agent"
    )
    model_path = str(ROOT_DIR / model_path)
    if Path(f"{model_path}.zip").exists():
        agents["PPO"] = PPOAgent.load(model_path, env)

    plot_episode_trace(env, agents, results_dir)
    print(f"\nAll plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
