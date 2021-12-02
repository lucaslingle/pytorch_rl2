"""
Implements the Bernoulli bandit environment from Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

import numpy as np

from rl2.envs.abstract import MetaEpisodicEnv


class BanditEnv(MetaEpisodicEnv):
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
        self.new_env()

    @property
    def max_episode_len(self):
        return 1

    @property
    def num_actions(self):
        """Get num_actions."""
        return self._num_actions

    def _new_payout_probabilities(self):
        """
        Samples a p_i ~ Uniform[0,1] to determine new payout probs for each arm.
        Returns:
            None
        """
        self._payout_probabilities = np.random.uniform(
            low=0.0, high=1.0, size=self._num_actions)

    def new_env(self) -> None:
        """
        Sample a new multi-armed bandit problem from distribution over problems.

        Returns:
            None
        """
        self._new_payout_probabilities()

    def reset(self) -> int:
        """
        Reset the environment. For MAB problems, the env is stateless,
        and this has no effect and is only included for compatibility
        with MetaEpisodicEnv abstract class.

        Returns:
            initial state.
        """
        self._state = 0
        return self._state

    def step(self, action, auto_reset=True) -> Tuple[int, float, bool, dict]:
        """
        Pull one arm of the multi-armed bandit, and observe one outcome.
        Args:
            action: action corresponding to an arm index.
            auto_reset: auto reset. if true, new_state will be from self.reset()

        Returns:
            new_state, reward, done, info.
        """

        # bernoulli bandit
        reward = np.random.binomial(
            n=1, p=self._payout_probabilities[action], size=1)[0]

        new_state = self._state
        done = True
        if done and auto_reset:
            new_state = self.reset()
        info = {}
        return new_state, reward, done, info
