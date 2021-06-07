"""
Implements MAB meta-reinforcement learning agent proposed by Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

import torch as tc

from rl2.agents.common import WeightNormedLinear, ValueHead, PolicyHead, one_hot


class BanditGRU(tc.nn.Module):
    """
    Bandit GRU from Duan et al., 2016.
    """
    def __init__(self, num_actions, feature_dim):
        super().__init__()
        self._num_actions = num_actions
        self._emb_dim = feature_dim
        self._input_dim = self._num_actions + 2
        self._hidden_dim = feature_dim

        self._x2z = WeightNormedLinear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            bias_initializer=tc.nn.init.zeros_)
        self._h2z = WeightNormedLinear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            bias_initializer=tc.nn.init.zeros_)

        self._x2r = WeightNormedLinear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            bias_initializer=tc.nn.init.zeros_)
        self._h2r = WeightNormedLinear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            bias_initializer=tc.nn.init.zeros_)

        self._x2hhat = WeightNormedLinear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            bias_initializer=tc.nn.init.zeros_)
        self._rh2hhat = WeightNormedLinear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.orthogonal_,
            bias_initializer=tc.nn.init.zeros_)

    def forward(self, prev_action, prev_reward, prev_done, prev_state):
        """
        Run recurrent state update and return new state.
        Args:
            prev_action: prev timestep action as tc.LongTensor w/ shape [B]
            prev_reward: prev timestep rew as tc.FloatTensor w/ shape [B]
            prev_done: prev timestep done flag as tc.FloatTensor w/ shape [B]
            prev_state: prev hidd state w/ shape [B, H].

        Returns:
            new_state.
        """
        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        in_vec = tc.cat((emb_a, prev_reward, prev_done), dim=-1)
        z = tc.nn.Sigmoid()(self._x2z(in_vec) + self._h2z(prev_state))
        r = tc.nn.Sigmoid()(self._x2r(in_vec) + self._h2r(prev_state))
        hhat = tc.nn.ReLU()(self._x2hhat(in_vec) +
                            self._rh2hhat(r * prev_state))
        h = (1. - z) * prev_state + z * hhat
        new_state = h

        return new_state


class ValueNetworkMAB(tc.nn.Module):
    """
    Value network from Duan et al., 2016 for multi-armed bandit problems.
    """
    def __init__(self, num_actions):
        super().__init__()
        self._num_actions = num_actions
        self._feature_dim = 256
        self._initial_state = tc.zeros(self._feature_dim)
        self._memory = BanditGRU(
            num_actions=self._num_actions,
            feature_dim=self._feature_dim)
        self._value_head = ValueHead(
            feature_dim=self._feature_dim)

    def initial_state(self, batch_size):  # pylint: disable=C0116
        return self._initial_state.unsqueeze(0).repeat(batch_size, 1)

    def forward(self, prev_action, prev_reward, prev_done, prev_state):  # pylint: disable=C0116
        new_state = self._memory(prev_action, prev_reward, prev_done, prev_state)
        v_pred = self._value_head(new_state)
        return v_pred, new_state


class PolicyNetworkMAB(tc.nn.Module):
    """
    Policy network from Duan et al., 2016 for multi-armed bandit problems.
    """
    def __init__(self, num_actions):
        super().__init__()
        self._num_actions = num_actions
        self._feature_dim = 256
        self._initial_state = tc.zeros(self._feature_dim)
        self._memory = BanditGRU(
            num_actions=self._num_actions,
            feature_dim=self._feature_dim)
        self._policy_head = PolicyHead(
            num_actions=self._num_actions,
            feature_dim=self._feature_dim)

    def initial_state(self, batch_size):  # pylint: disable=C0116
        return self._initial_state.unsqueeze(0).repeat(batch_size, 1)

    def forward(self, prev_action, prev_reward, prev_done, prev_state):  # pylint: disable=C0116
        new_state = self._memory(prev_action, prev_reward, prev_done, prev_state)
        pi_dist = self._policy_head(new_state)
        return pi_dist, new_state
