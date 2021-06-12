"""
Implements abstract class for stateful meta-reinforcement learning agents.
"""

from typing import Union, Tuple
import abc

import torch as tc


class StatefulPolicyNet(abc.ABC, tc.nn.Module):
    @abc.abstractmethod
    def initial_state(self, batch_size: int) -> tc.FloatTensor:
        """
        Returns a batch-tiled initial state of zeros.

        Args:
            batch_size: batch size to tile the initial state by.

        Returns:
            initial_state FloatTensor.
        """
        pass

    @abc.abstractmethod
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
                or tc.FloatTensor with shape [B].
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep rew as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev hidd state w/ shape [B, H].

        Returns:
            A tuple containing the parametrized policy's action distribution
              and the new state of the stateful policy.
        """
        pass


class StatefulValueNet(abc.ABC, tc.nn.Module):
    @abc.abstractmethod
    def initial_state(self, batch_size: int) -> tc.FloatTensor:
        """
        Returns a batch-tiled initial state of zeros.

        Args:
            batch_size: batch size to tile the initial state by.

        Returns:
            initial_state FloatTensor.
        """
        pass

    @abc.abstractmethod
    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Runs recurrent state update and returns value estimate and new state.

        Args:
            curr_obs: current timestep observation as tc.LongTensor
                or tc.FloatTensor with shape [B].
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep rew as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev hidd state w/ shape [B, H].

        Returns:
            A tuple containing the parametrized value function's value estimate
              and the new state of the stateful value function.
        """
        pass
