"""
Implements preprocessing for vision-based MDPs/POMDPs.
"""

import abc

import torch as tc

from rl2.agents_v2.preprocessing.common import one_hot


class AbstractVisionNet(abc.ABC, tc.nn.Module):
    """
    Abstract class implementing a vision network
    """
    def forward(self, curr_obs: tc.FloatTensor) -> tc.FloatTensor:
        """
        Embeds visual observations into feature vectors.

        Args:
            curr_obs: tc.FloatTensor of shape [B, C, H, W]

        Returns:
            a tc.FloatTensor of shape [B, F]
        """
        pass


class MDPPreprocessing(tc.nn.Module):
    def __init__(self, num_actions: int, vision_net: AbstractVisionNet):
        super().__init__()
        self._num_actions = num_actions
        self._vision_net = vision_net

    def forward(
        self,
        curr_obs: tc.FloatTensor,
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
            tc.FloatTensor of shape [B, ..., F+A+2]
        """

        curr_obs_shape = list(curr_obs.shape)
        curr_obs = curr_obs.view(-1, *curr_obs_shape[-3:])
        emb_o = self._vision_net(curr_obs)
        emb_o = emb_o.view(*curr_obs_shape[:-3], emb_o.shape[-1])

        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat(
            (emb_o, emb_a, prev_reward, prev_done), dim=-1).float()
        return vec
