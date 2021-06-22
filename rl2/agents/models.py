"""
Stateful networks for meta-reinforcement learning.
"""

from typing import Tuple

import torch as tc

from rl2.agents.common import WeightNormedLinear


class DuanGRU(tc.nn.Module):
    """
    GRU from Duan et al., 2016.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

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

    def forward(
        self,
        input_vec: tc.LongTensor,
        prev_state: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Run recurrent state update.
        Args:
            input_vec: current timestep input vector as tc.FloatTensor
            prev_state: prev hidd state w/ shape [B, H].

        Returns:
            new_state.
        """
        z = tc.nn.Sigmoid()(self._x2z(input_vec) + self._h2z(prev_state))
        r = tc.nn.Sigmoid()(self._x2r(input_vec) + self._h2r(prev_state))
        hhat = tc.nn.ReLU()(
            self._x2hhat(input_vec) + self._rh2hhat(r * prev_state))
        h = (1. - z) * prev_state + z * hhat
        new_state = h

        return new_state


class LSTM(tc.nn.Module):
    """
    LSTM.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        self._linear = tc.nn.Linear(
            (self._input_dim + self._hidden_dim), 4 * self._hidden_dim)

        tc.nn.init.uniform_(self._linear.weight, -0.10, 0.10),
        tc.nn.init.zeros_(self._linear.bias)

    def forward(
        self,
        input_vec: tc.LongTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Run recurrent state update.
        Args:
            input_vec: current timestep input vector as tc.FloatTensor
            prev_state: prev lstm state w/ shape [B, 2*H].

        Returns:
            new_cell_state, new_hidden_state.
        """
        hidden_vec, c_prev = tc.chunk(prev_state, 2, dim=-1)
        vec = tc.cat((input_vec, hidden_vec), dim=-1)
        fioj = self._linear(vec)
        f, i, o, j = tc.chunk(fioj, 4, dim=-1)
        f = tc.nn.Sigmoid()(f + 1.0)
        i = tc.nn.Sigmoid()(i)
        o = tc.nn.Sigmoid()(o)
        j = tc.nn.Tanh()(j)
        c_new = f * c_prev + i * j
        h_new = o * c_new
        return h_new, c_new


def lstm_postprocessing(
        hidden_state: tc.FloatTensor,
        cell_state: tc.FloatTensor
    ) -> tc.FloatTensor:
        state = tc.cat((hidden_state, cell_state), dim=-1).float()
        return state
