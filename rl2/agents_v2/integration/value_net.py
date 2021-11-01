"""
Implements StatefulValueNet module
"""

from typing import Union, Tuple, Optional

import torch as tc


class StatefulValueNet(tc.nn.Module):
    def __init__(self, preprocessing, architecture, value_head):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._value_head = value_head

    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: Optional[tc.FloatTensor]
    ) -> Tuple[tc.distributions.Categorical, tc.FloatTensor]:
        """
        Runs agent state update and returns value estimate(s) and new state.

        Args:
            curr_obs: current observation(s) tensor with shape [B, ..., ?].
            prev_action: previous action(s) tensor with shape [B, ...]
            prev_reward: previous rewards(s) tensor with shape [B, ...]
            prev_done: previous done flag(s) tensor with shape [B, ...]
            prev_state: agent's previous state.

        Notes:
            '...' must be either one dimensional or must not exist;
            for recurrent policies, it should not exist;
            for attentive policies, it should be the length of presents.

        Returns:
            Tuple containing value estimate(s) with batch shape [B, ...]
               and agent state tc.FloatTensor with batch shape [B, ...+].
        """
        vec = self._preprocessing(
            curr_obs, prev_action, prev_reward, prev_done)

        features, new_state = self._architecture(
            input_vec=vec,
            prev_state=prev_state)

        vpred = self._value_head(features)

        return vpred, new_state
