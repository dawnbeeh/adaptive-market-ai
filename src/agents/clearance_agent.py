import numpy as np


class ClearanceAgent:

    def __init__(self, env):
        self.env = env

    def predict(self, observation):
        """
        observation format:
        [inventory, freshness, market_price, demand, storage_cost, time_step]
        """
        inventory = observation[0]
        freshness = observation[1]
        demand = observation[3]
        time_step = observation[5]

        max_steps = getattr(self.env, "max_steps", 50)
        time_ratio = time_step / max_steps if max_steps > 0 else 0.0

        if inventory <= 0:
            return np.array([0.0, 1.0], dtype=np.float32), None

        # High freshness, early/mid episode
        if freshness >= 0.75:
            if demand >= 0.7:
                sell_ratio = 0.55
                price_multiplier = 1.10
            else:
                sell_ratio = 0.40
                price_multiplier = 1.20

        # Medium freshness
        elif freshness >= 0.40:
            if time_ratio < 0.6:
                sell_ratio = 0.65
                price_multiplier = 0.95
            else:
                sell_ratio = 0.80
                price_multiplier = 0.85

        # Low freshness: start clearance
        else:
            if time_ratio < 0.8:
                sell_ratio = 0.90
                price_multiplier = 0.75
            else:
                sell_ratio = 1.00
                price_multiplier = 0.60

        # Extra urgency near the end
        if time_ratio >= 0.9:
            sell_ratio = 1.00
            price_multiplier = min(price_multiplier, 0.60)

        action = np.array([sell_ratio, price_multiplier], dtype=np.float32)
        return action, None