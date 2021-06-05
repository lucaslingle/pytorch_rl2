"""
Implements the Bernoulli bandit environment from Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

import numpy as np


class BanditEnv:
    """
    An environment where each step is a pull from a multi-armed bandit.
    The bandit is a stationary bernoulli bandit.
    The environment can be reset so that a new MAB replaces the old one,
        thus permitting memory-augmented agents to meta-learn exploration
        strategies that generalize across environments in the distribution.
    """
    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._state = None
        self._payout_probabilities = None
        self.new_bandit()

    @property
    def num_actions(self):
        """Get num_actions."""
        return self._num_actions

    def new_bandit(self) -> None:
        """
        Sample a new multi-armed bandit problem from distribution over problems.

        Returns:
            None
        """
        self._new_payout_probabilities()

    def _new_payout_probabilities(self):
        """
        Samples a p_i ~ Uniform[0,1] to determine new payout probs for each arm.
        Returns:
            None
        """
        self._payout_probabilities = np.random.uniform(
            low=0.0, high=1.0, size=self._num_actions)

    def step(self, a):
        """
        Pull one arm of the multi-armed bandit, and observe one outcome.
        Args:
            a: action corresponding to an arm index.

        Returns:
            new_state, reward, done, info.

        Note that new_state is always None, done is always True,
        and info is always an empty dictionary.
        """
        reward = np.random.binomial(
            n=1, p=self._payout_probabilities[a], size=1)[0]  # bernoulli bandit
        new_state = self._state
        done = True
        info = {}
        return new_state, reward, done, info
