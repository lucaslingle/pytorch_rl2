"""
Implements LSTM for RL^2.
"""

from typing import Tuple

import torch as tc

from rl2.agents_v2.architectures.common import LayerNorm


class LSTM(tc.nn.Module):
    def __init__(
        self, input_dim, hidden_dim, forget_bias=1.0, use_ln=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._use_ln = use_ln
        self._forget_bias = forget_bias

        self._x2fioj = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=(4 * self._hidden_dim),
            bias=(not self._use_ln))
        tc.nn.init.uniform_(self._x2fioj.weight, -0.10, 0.10)

        self._h2fioj = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=(4 * self._hidden_dim),
            bias=False)
        tc.nn.init.uniform_(self._x2fioj.weight, -0.10, 0.10)

        if self._use_ln:
            self._x2fioj_ln = LayerNorm(units=(4 * self._hidden_dim))
            self._h2fioj_ln = LayerNorm(units=(4 * self._hidden_dim))
            self._c_out_ln = LayerNorm(units=self._hidden_dim)

        self._initial_state = tc.zeros(2 * self._hidden_dim)

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
        input_vec: tc.FloatTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Run recurrent state update.
        Args:
            input_vec: current timestep input vector as tc.FloatTensor
            prev_state: prev lstm state w/ shape [B, 2*H].
        Returns:
            features, new_state.
        """
        hidden_vec, c_prev = tc.chunk(prev_state, 2, dim=-1)
        fioj_from_x = self._x2fioj(input_vec)
        fioj_from_h = self._h2fioj(hidden_vec)
        if self._use_ln:
            fioj_from_x = self._x2fioj_ln(fioj_from_x)
            fioj_from_h = self._h2fioj_ln(fioj_from_h)
        fioj = fioj_from_x + fioj_from_h
        f, i, o, j = tc.chunk(fioj, 4, dim=-1)
        f = tc.nn.Sigmoid()(f + self._forget_bias)
        i = tc.nn.Sigmoid()(i)
        o = tc.nn.Sigmoid()(o)
        j = tc.nn.ReLU()(j)
        c_new = f * c_prev + i * j
        h_new = o * (self._c_out_ln(c_new) if self._use_ln else c_new)

        features = h_new
        new_state = tc.cat((h_new, c_new), dim=-1).float()
        return features, new_state
