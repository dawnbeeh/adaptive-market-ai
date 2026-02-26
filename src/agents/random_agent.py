"""Random baseline agent."""

import numpy as np


class RandomAgent:
    """Samples uniform random actions each step."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        return self.env.action_space.sample(), None
