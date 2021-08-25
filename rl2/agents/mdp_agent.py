"""
Implements MDP meta-reinforcement learning agent proposed by Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

import torch as tc

from rl2.agents.abstract import StatefulPolicyNet, StatefulValueNet
from rl2.agents.common import (
    one_hot,
    lstm_postprocessing,
    DuanGRU,
    LSTM,
    PolicyHead,
    ValueHead
)


def mdp_preprocessing(
        num_states: int,
        num_actions: int,
        curr_obs: tc.LongTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:

        emb_o = one_hot(curr_obs, depth=num_states)
        emb_a = one_hot(prev_action, depth=num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        input_vec = tc.cat(
            (emb_o, emb_a, prev_reward, prev_done), dim=-1).float()
        return input_vec


class PolicyNetworkGRU(StatefulPolicyNet):
    """
    Policy network from Duan et al., 2016 for MDPs.
    """
    def __init__(
            self, num_states, num_actions, num_features, use_wn, use_ln,
            forget_bias, reset_after=True
    ):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_features = num_features
        self._use_wn = use_wn
        self._use_ln = use_ln
        self._forget_bias = forget_bias
        self._reset_after = reset_after
        self._initial_state = tc.zeros(self._num_features)
        self._memory = DuanGRU(
            input_dim=self._num_states+self._num_actions+2,
            hidden_dim=self._num_features,
            use_wn=self._use_wn,
            use_ln=self._use_ln,
            forget_bias=self._forget_bias,
            reset_after=self._reset_after)

        self._policy_head = PolicyHead(
            num_features=self._num_features,
            num_actions=self._num_actions,
            use_wn=self._use_wn)

    def initial_state(self, batch_size: int) -> tc.FloatTensor:
        """
        Return initial state of zeros.
        Args:
            batch_size: batch size to tile the initial state by.
        Returns:
            initial_state FloatTensor.
        """
        return self._initial_state.unsqueeze(0).repeat(batch_size, 1)

    def forward(
        self,
        curr_obs: tc.LongTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.distributions.Categorical, tc.FloatTensor]:
        """
        Runs recurrent state update and returns policy dist and new state.

        Args:
            curr_obs: current timestep observation as tc.LongTensor w/ shape [B]
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep reward as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev hidden state w/ shape [B, H].

        Returns:
            A tuple containing the parametrized policy's action distribution
              and the new state of the stateful policy.
        """
        input_vec = mdp_preprocessing(
            num_states=self._num_states,
            num_actions=self._num_actions,
            curr_obs=curr_obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=prev_done)

        new_state = self._memory(
            input_vec=input_vec,
            prev_state=prev_state)

        pi_dist = self._policy_head(
            features=new_state)

        return pi_dist, new_state


class PolicyNetworkLSTM(StatefulPolicyNet):
    """
    LSTM policy network for meta-reinforcement learning over MDPs.
    """
    def __init__(self, num_states, num_actions, num_features, use_wn, use_ln):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_features = num_features
        self._use_wn = use_wn
        self._use_ln = use_ln
        self._initial_state = tc.zeros(2 * self._num_features)
        self._memory = LSTM(
            input_dim=self._num_states+self._num_actions+2,
            hidden_dim=self._num_features,
            use_wn=self._use_wn,
            use_ln=self._use_ln)

        self._policy_head = PolicyHead(
            num_actions=self._num_actions,
            num_features=self._num_features,
            use_wn=self._use_wn)

    def initial_state(self, batch_size: int) -> tc.FloatTensor:
        """
        Return initial state of zeros.
        Args:
            batch_size: batch size to tile the initial state by.
        Returns:
            initial_state FloatTensor.
        """
        return self._initial_state.unsqueeze(0).repeat(batch_size, 1)

    def forward(
        self,
        curr_obs: tc.LongTensor,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.distributions.Categorical, tc.FloatTensor]:
        """
        Runs recurrent state update and returns policy dist and new state.

        Args:
            curr_obs: current timestep observation as tc.LongTensor w/ shape [B]
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep reward as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev lstm state w/ shape [B, 2*H].

        Returns:
            A tuple containing the parametrized policy's action distribution
              and the new state of the stateful policy.
        """
        input_vec = mdp_preprocessing(
            num_states=self._num_states,
            num_actions=self._num_actions,
            curr_obs=curr_obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=prev_done)

        new_hidden_state, new_cell_state = self._memory(
            input_vec=input_vec,
            prev_state=prev_state)

        pi_dist = self._policy_head(
            features=new_hidden_state)

        new_state = lstm_postprocessing(
            hidden_state=new_hidden_state,
            cell_state=new_cell_state)

        return pi_dist, new_state


class ValueNetworkGRU(StatefulValueNet):
    """
    Value network from Duan et al., 2016 for MDPs.
    """
    def __init__(
            self, num_states, num_actions, num_features, use_wn, use_ln,
            forget_bias, reset_after=True
    ):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_features = num_features
        self._use_wn = use_wn
        self._use_ln = use_ln
        self._forget_bias = forget_bias
        self._reset_after = reset_after
        self._initial_state = tc.zeros(self._num_features)
        self._memory = DuanGRU(
            input_dim=self._num_states+self._num_actions+2,
            hidden_dim=self._num_features,
            use_wn=self._use_wn,
            use_ln=self._use_ln,
            forget_bias=self._forget_bias,
            reset_after=self._reset_after)

        self._value_head = ValueHead(
            num_features=self._num_features,
            use_wn=self._use_wn)

    def initial_state(self, batch_size: int) -> tc.FloatTensor:
        """
        Return initial state of zeros.
        Args:
            batch_size: batch size to tile the initial state by.
        Returns:
            initial_state FloatTensor.
        """
        return self._initial_state.unsqueeze(0).repeat(batch_size, 1)

    def forward(
            self,
            curr_obs: tc.LongTensor,
            prev_action: tc.LongTensor,
            prev_reward: tc.FloatTensor,
            prev_done: tc.FloatTensor,
            prev_state: tc.FloatTensor
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Runs recurrent state update and returns value estimate and new state.

        Args:
            curr_obs: current timestep observation as tc.LongTensor w/ shape [B]
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep reward as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev hidden state w/ shape [B, H].

        Returns:
            A tuple containing the parametrized value function's value estimate
              and the new state of the stateful value function.
        """
        input_vec = mdp_preprocessing(
            num_states=self._num_states,
            num_actions=self._num_actions,
            curr_obs=curr_obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=prev_done)

        new_state = self._memory(
            input_vec=input_vec,
            prev_state=prev_state)

        v_pred = self._value_head(
            features=new_state)

        return v_pred, new_state


class ValueNetworkLSTM(StatefulValueNet):
    """
    LSTM value network for meta-reinforcement learning in MDPs.
    """
    def __init__(self, num_states, num_actions, num_features, use_wn, use_ln):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_features = num_features
        self._use_wn = use_wn
        self._use_ln = use_ln
        self._initial_state = tc.zeros(2 * self._num_features)
        self._memory = LSTM(
            input_dim=self._num_states+self._num_actions+2,
            hidden_dim=self._num_features,
            use_wn=self._use_wn,
            use_ln=self._use_ln)

        self._value_head = ValueHead(
            num_features=self._num_features,
            use_wn=self._use_wn)

    def initial_state(self, batch_size: int) -> tc.FloatTensor:
        """
        Return initial state of zeros.
        Args:
            batch_size: batch size to tile the initial state by.
        Returns:
            initial_state FloatTensor.
        """
        return self._initial_state.unsqueeze(0).repeat(batch_size, 1)

    def forward(
            self,
            curr_obs: tc.LongTensor,
            prev_action: tc.LongTensor,
            prev_reward: tc.FloatTensor,
            prev_done: tc.FloatTensor,
            prev_state: tc.FloatTensor
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Runs recurrent state update and returns value estimate and new state.

        Args:
            curr_obs: current timestep observation as tc.LongTensor w/ shape [B]
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep reward as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev lstm state w/ shape [B, 2*H].

        Returns:
            A tuple containing the parametrized value function's value estimate
              and the new state of the stateful value function.
        """
        input_vec = mdp_preprocessing(
            num_states=self._num_states,
            num_actions=self._num_actions,
            curr_obs=curr_obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_done=prev_done)

        new_hidden_state, new_cell_state = self._memory(
            input_vec=input_vec,
            prev_state=prev_state)

        v_pred = self._value_head(
            features=new_hidden_state)

        new_state = lstm_postprocessing(
            hidden_state=new_hidden_state,
            cell_state=new_cell_state)

        return v_pred, new_state
