import numpy as np


class GreedyAgent:

    def __init__(self):
        # candidate action grid
        self.sell_ratios = np.linspace(0.0, 1.0, 5)   # [0, .25, .5, .75, 1]
        self.price_multipliers = np.linspace(0.5, 2.0, 6)

    def predict(self, observation):
        """
        observation format:
        [inventory, freshness, market_price, demand, storage_cost, time_step]
        """

        inventory = observation[0]
        freshness = observation[1]
        market_price = observation[2]
        demand = observation[3]
        storage_cost = observation[4]

        best_reward = -np.inf
        best_action = np.array([0.0, 1.0])

        for sell_ratio in self.sell_ratios:
            for price_multiplier in self.price_multipliers:

                units_listed = sell_ratio * inventory
                ask_price = price_multiplier * market_price

                # approximate demand-limited sales
                units_sold = min(units_listed, demand)

                # freshness-adjusted price
                effective_price = ask_price * freshness

                revenue = units_sold * effective_price

                # approximate storage cost for remaining inventory
                remaining_inventory = inventory - units_sold
                cost = remaining_inventory * storage_cost

                reward = revenue - cost

                if reward > best_reward:
                    best_reward = reward
                    best_action = np.array([sell_ratio, price_multiplier])

        return best_action, None