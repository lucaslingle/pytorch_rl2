"""
Implements preprocessing for tabular MABs and MDPs.
"""

import torch as tc

from rl2.agents.preprocessing.common import one_hot, Preprocessing


class MABPreprocessing(Preprocessing):
    def __init__(self, num_actions: int):
        super().__init__()
        self._num_actions = num_actions

    @property
    def output_dim(self):
        return self._num_actions + 2

    def forward(
        self,
        curr_obs: tc.LongTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Creates an input vector for a meta-learning agent.

        Args:
            curr_obs: tc.LongTensor of shape [B, ...]; will be ignored.
            prev_action: tc.LongTensor of shape [B, ...]
            prev_reward: tc.FloatTensor of shape [B, ...]
            prev_done: tc.FloatTensor of shape [B, ...]

        Returns:
            tc.FloatTensor of shape [B, ..., A+2]
        """

        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat((emb_a, prev_reward, prev_done), dim=-1).float()
        return vec


class MDPPreprocessing(Preprocessing):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions

    @property
    def output_dim(self):
        return self._num_states + self._num_actions + 2

    def forward(
        self,
        curr_obs: tc.LongTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Creates an input vector for a meta-learning agent.

        Args:
            curr_obs: tc.FloatTensor of shape [B, ..., C, H, W]
            prev_action: tc.LongTensor of shape [B, ...]
            prev_reward: tc.FloatTensor of shape [B, ...]
            prev_done: tc.FloatTensor of shape [B, ...]

        Returns:
            tc.FloatTensor of shape [B, ..., S+A+2]
        """

        emb_o = one_hot(curr_obs, depth=self._num_states)
        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat(
            (emb_o, emb_a, prev_reward, prev_done), dim=-1).float()
        return vec
