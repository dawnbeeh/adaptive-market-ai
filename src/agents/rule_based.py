"""Rule-based heuristic agent."""

import numpy as np


class RuleBasedAgent:
    """Heuristic strategy: discount when freshness is low, sell more when demand is high."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        inventory = obs[0]
        freshness = obs[1]
        market_price = obs[2]
        demand_level = obs[3]
        time_norm = obs[5]

        # Sell ratio: sell more aggressively when freshness is low or time is running out
        sell_ratio = 0.3  # base
        if freshness < 0.5:
            sell_ratio += 0.3
        if freshness < 0.2:
            sell_ratio += 0.3
        if time_norm > 0.7:
            sell_ratio += 0.2

        # Also sell more when demand is high
        if demand_level > 60:
            sell_ratio += 0.1

        sell_ratio = np.clip(sell_ratio, 0.0, 1.0)

        # Price multiplier: charge more when fresh, discount when stale
        price_multiplier = 0.8 + 0.8 * freshness  # range ~[0.8, 1.6]

        # Discount further when time is running out
        if time_norm > 0.8:
            price_multiplier *= 0.8

        price_multiplier = np.clip(price_multiplier, 0.5, 2.0)

        action = np.array([sell_ratio, price_multiplier], dtype=np.float32)
        return action, None
