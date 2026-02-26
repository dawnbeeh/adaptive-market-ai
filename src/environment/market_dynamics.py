"""Market simulation models for the perishable goods environment."""

import numpy as np


class PriceModel:
    """Random walk with drift and mean reversion."""

    def __init__(self, base_price: float, drift: float = 0.0,
                 volatility: float = 0.3, mean_reversion: float = 0.1):
        self.base_price = base_price
        self.drift = drift
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.price = base_price

    def reset(self, rng: np.random.Generator):
        self.price = self.base_price
        return self.price

    def step(self, rng: np.random.Generator) -> float:
        reversion = self.mean_reversion * (self.base_price - self.price)
        shock = self.volatility * rng.standard_normal()
        self.price += self.drift + reversion + shock
        self.price = max(self.price, 0.01)
        return self.price


class DemandModel:
    """Sinusoidal seasonality + noise + price elasticity."""

    def __init__(self, base: float = 50.0, amplitude: float = 20.0,
                 period: float = 7.0, noise_std: float = 5.0,
                 price_elasticity: float = 1.5):
        self.base = base
        self.amplitude = amplitude
        self.period = period
        self.noise_std = noise_std
        self.price_elasticity = price_elasticity

    def get_demand(self, time_step: int, price_ratio: float,
                   rng: np.random.Generator) -> float:
        seasonal = self.amplitude * np.sin(2 * np.pi * time_step / self.period)
        noise = self.noise_std * rng.standard_normal()
        raw_demand = self.base + seasonal + noise
        # Higher price ratio -> lower demand
        demand = raw_demand * (1.0 / price_ratio) ** self.price_elasticity
        return max(demand, 0.0)


class DecayModel:
    """Exponential freshness decay."""

    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate

    def decay(self, freshness: float) -> float:
        new_freshness = freshness * np.exp(-self.decay_rate)
        return max(new_freshness, 0.0)


class ShockGenerator:
    """Random market shocks: demand spikes or supply disruptions."""

    def __init__(self, probability: float = 0.05, magnitude: float = 0.5):
        self.probability = probability
        self.magnitude = magnitude

    def generate(self, rng: np.random.Generator) -> dict:
        shock = {"demand_multiplier": 1.0, "price_shock": 0.0}
        if rng.random() < self.probability:
            if rng.random() < 0.5:
                # Demand spike
                shock["demand_multiplier"] = 1.0 + self.magnitude
            else:
                # Supply disruption -> price increase
                shock["price_shock"] = self.magnitude * 2.0
        return shock
