"""
Implements StatefulValueNet class.
"""

from typing import Union, Tuple, Optional, TypeVar, Generic

import torch as tc


ArchitectureState = TypeVar('ArchitectureState')


class StatefulValueNet(tc.nn.Module, Generic[ArchitectureState]):
    def __init__(self, preprocessing, architecture, value_head):
        super().__init__()
        self._preprocessing = preprocessing
        self._architecture = architecture
        self._value_head = value_head

    def initial_state(self, batch_size: int) -> Optional[ArchitectureState]:
        return self._architecture.initial_state(batch_size=batch_size)

    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: Optional[ArchitectureState]
    ) -> Tuple[tc.FloatTensor, ArchitectureState]:
        """
        Runs preprocessing and the architecture's state update;
        returns value estimate(s) and new state.

        Args:
            curr_obs: current observation(s) tensor with shape [B, ..., ?].
            prev_action: previous action(s) tensor with shape [B, ...]
            prev_reward: previous rewards(s) tensor with shape [B, ...]
            prev_done: previous done flag(s) tensor with shape [B, ...]
            prev_state: the architecture's previous state.

        Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            Tuple containing value estimate(s) with batch shape [B, ...]
               and the architecture's new state.
        """
        inputs = self._preprocessing(
            curr_obs, prev_action, prev_reward, prev_done)

        features, new_state = self._architecture(
            inputs=inputs, prev_state=prev_state)

        vpred = self._value_head(features)

        return vpred, new_state
