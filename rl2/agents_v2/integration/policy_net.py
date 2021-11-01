"""
Implements StatefulPolicyNet module
"""

from typing import Union, Tuple

import torch as tc


class StatefulPolicyNet(tc.nn.Module):
    def __init__(self, preprocessing, architecture, policy_head):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._policy_head = policy_head

    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.distributions.Categorical, tc.FloatTensor]:
        """
        Runs recurrent state update and returns policy dist and new state.

        Args:
            curr_obs: current timestep observation as tc.LongTensor
                or tc.FloatTensor with shape [B, ..., ?].
            prev_action: prev timestep actions as tc.LongTensor w/ shape [B, ...]
            prev_reward: prev timestep rews as tc.FloatTensor w/ shape [B, ...]
            prev_done: prev timestep done flags as tc.FloatTensor w/ shape [B, ...]
            prev_state: prev agent state.

        Returns:
            Tuple containing policy distribution(s) with batch shape [B, ...]
               and agent state tc.FloatTensor with batch shape [B, ...+].
        """
        vec = self._preprocessing(
            curr_obs, prev_action, prev_reward, prev_done)

        features, new_state = self._architecture(
            input_vec=vec,
            prev_state=prev_state)

        dists = self._policy_head(features)

        return dists, new_state