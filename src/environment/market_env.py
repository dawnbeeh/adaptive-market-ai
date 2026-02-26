"""Perishable goods market Gymnasium environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.market_dynamics import (
    DecayModel,
    DemandModel,
    PriceModel,
    ShockGenerator,
)


class PerishableMarketEnv(gym.Env):
    """RL environment for selling perishable goods in a dynamic market.

    Observation (6 dims):
        0: inventory       - current stock level [0, initial_inventory]
        1: freshness       - product freshness [0, 1]
        2: market_price    - current market price [0, inf)
        3: demand_level    - current demand [0, inf)
        4: storage_cost    - accumulated storage cost [0, inf)
        5: time_step       - normalized day in episode [0, 1]

    Action (2 dims):
        0: sell_ratio        - fraction of inventory to list [0, 1]
        1: price_multiplier  - price relative to market [0.5, 2.0]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        env_cfg = cfg.get("environment", {})
        mkt_cfg = cfg.get("market", {})

        self.max_steps = env_cfg.get("max_steps", 30)
        self.initial_inventory = env_cfg.get("initial_inventory", 100.0)
        self.initial_freshness = env_cfg.get("initial_freshness", 1.0)
        self.storage_cost_per_unit = env_cfg.get("storage_cost_per_unit", 0.5)
        self.spoilage_penalty_per_unit = env_cfg.get("spoilage_penalty_per_unit", 2.0)
        self.base_price = env_cfg.get("base_price", 10.0)

        self.price_model = PriceModel(
            base_price=self.base_price,
            drift=mkt_cfg.get("price_drift", 0.0),
            volatility=mkt_cfg.get("price_volatility", 0.3),
            mean_reversion=mkt_cfg.get("mean_reversion_rate", 0.1),
        )
        self.demand_model = DemandModel(
            base=mkt_cfg.get("demand_base", 50.0),
            amplitude=mkt_cfg.get("demand_amplitude", 20.0),
            period=mkt_cfg.get("demand_period", 7.0),
            noise_std=mkt_cfg.get("demand_noise_std", 5.0),
            price_elasticity=mkt_cfg.get("price_elasticity", 1.5),
        )
        self.decay_model = DecayModel()
        self.shock_gen = ShockGenerator(
            probability=mkt_cfg.get("shock_probability", 0.05),
            magnitude=mkt_cfg.get("shock_magnitude", 0.5),
        )

        # Observation: inventory, freshness, market_price, demand, storage_cost, time
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 1.0, np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
        )

        # Action: sell_ratio [0,1], price_multiplier [0.5, 2.0]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.5], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
        )

        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self):
        self.inventory = self.initial_inventory
        self.freshness = self.initial_freshness
        self.market_price = self.base_price
        self.demand_level = 0.0
        self.cumulative_storage_cost = 0.0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_state()
        self.market_price = self.price_model.reset(self._rng)
        self.demand_level = self.demand_model.get_demand(
            0, 1.0, self._rng
        )
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        sell_ratio = float(action[0])
        price_multiplier = float(action[1])

        # Generate shocks
        shock = self.shock_gen.generate(self._rng)

        # Update market price
        self.market_price = self.price_model.step(self._rng)
        self.market_price += shock["price_shock"]
        self.market_price = max(self.market_price, 0.01)

        # Calculate listing
        units_listed = sell_ratio * self.inventory
        ask_price = price_multiplier * self.market_price

        # Get demand at this price
        self.demand_level = self.demand_model.get_demand(
            self.current_step, price_multiplier, self._rng
        )
        self.demand_level *= shock["demand_multiplier"]

        # Units actually sold = min(listed, demand)
        units_sold = min(units_listed, self.demand_level)

        # Freshness affects willingness to pay — discount revenue by freshness
        effective_price = ask_price * self.freshness
        revenue = units_sold * effective_price

        # Update inventory
        self.inventory -= units_sold

        # Storage cost on remaining inventory
        storage_cost = self.inventory * self.storage_cost_per_unit
        self.cumulative_storage_cost += storage_cost

        # Freshness decay
        self.freshness = self.decay_model.decay(self.freshness)

        # Spoilage: if freshness drops below threshold, lose some inventory
        spoilage_loss = 0.0
        if self.freshness < 0.1:
            spoiled = self.inventory * 0.2
            self.inventory -= spoiled
            spoilage_loss = spoiled
        self.current_step += 1

        # Reward
        reward = revenue - storage_cost - spoilage_loss

        # Termination conditions
        terminated = self.inventory <= 0.0
        truncated = self.current_step >= self.max_steps

        # End-of-episode penalty for unsold inventory
        if (terminated or truncated) and self.inventory > 0:
            reward -= self.inventory * self.spoilage_penalty_per_unit * 0.5

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.inventory,
            self.freshness,
            self.market_price,
            self.demand_level,
            self.cumulative_storage_cost,
            self.current_step / self.max_steps,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "inventory": self.inventory,
            "freshness": self.freshness,
            "market_price": self.market_price,
            "demand_level": self.demand_level,
            "storage_cost": self.cumulative_storage_cost,
            "step": self.current_step,
        }
