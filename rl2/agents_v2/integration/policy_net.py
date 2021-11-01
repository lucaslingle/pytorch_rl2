"""
Implements StatefulPolicyNet class.
"""

from typing import Union, Tuple, Optional

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
        prev_state: Optional[tc.FloatTensor]
    ) -> Tuple[tc.distributions.Categorical, tc.FloatTensor]:
        """
        Runs preprocessing and the architecture's state update;
        returns policy distribution(s) and new state.

        Args:
            curr_obs: current observation(s) tensor with shape [B, ..., ?].
            prev_action: previous action(s) tensor with shape [B, ...]
            prev_reward: previous rewards(s) tensor with shape [B, ...]
            prev_done: previous done flag(s) tensor with shape [B, ...]
            prev_state: the architecture's previous state.

        Notes:
            '...' must be either one dimensional or must not exist;
            for recurrent architectures, it should not exist;
            for attentive architectures, it should be the length of presents.

        Returns:
            Tuple containing policy distribution(s) with batch shape [B, ...]
               and the architecture's new state.
        """
        vec = self._preprocessing(
            curr_obs, prev_action, prev_reward, prev_done)

        features, new_state = self._architecture(
            input_vec=vec,
            prev_state=prev_state)

        dist = self._policy_head(features)

        return dist, new_state
