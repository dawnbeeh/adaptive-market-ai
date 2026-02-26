# Adaptive Market AI

A reinforcement learning system that learns optimal selling and pricing strategies for perishable goods in a dynamic market environment.

## Overview

In markets dealing with perishable assets, operators must continuously decide **when to sell**, **how much to sell**, and **at what price** — all while asset values decay, demand shifts, and unexpected market events occur. This project builds a custom simulation environment and trains an RL agent (PPO) to maximize long-term profit under these conditions.

## Key Features

- **Custom Gymnasium Environment** — Simulates a perishable goods market with dynamic pricing, stochastic demand, spoilage mechanics, and random shock events
- **PPO Agent** — Learns adaptive selling and pricing strategies through interaction with the environment
- **Baseline Comparisons** — Rule-based heuristic and random agents for benchmarking
- **Analysis & Visualization** — Training curves, strategy analysis, and shock-response evaluation

## Project Structure

```
adaptive-market-ai/
├── configs/              # Environment & training hyperparameters
├── src/
│   ├── environment/      # Custom Gymnasium market environment
│   ├── agents/           # PPO, rule-based, and random agents
│   ├── training/         # Training scripts
│   └── evaluation/       # Evaluation and visualization tools
├── results/              # Saved models, logs, and figures
└── tests/                # Unit tests
```

## Installation

```bash
git clone https://github.com/<username>/adaptive-market-ai.git
cd adaptive-market-ai
pip install -r requirements.txt
```

## Usage

### Train the PPO Agent
```bash
python -m src.training.train
```

### Evaluate & Compare Agents
```bash
python -m src.evaluation.evaluate
```

### Generate Visualizations
```bash
python -m src.evaluation.visualize
```

## Tech Stack

- Python 3.10+
- [Gymnasium](https://gymnasium.farama.org/) — RL environment framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO implementation
- NumPy, Pandas, Matplotlib