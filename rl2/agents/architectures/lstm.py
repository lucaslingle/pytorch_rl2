"""
Implements LSTM for RL^2.
"""

from typing import Tuple

import torch as tc

from rl2.agents.architectures.common.normalization import LayerNorm


class LSTM(tc.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        forget_bias=1.0,
        use_ln=True
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
        tc.nn.init.xavier_normal_(self._x2fioj.weight)
        if not self._use_ln:
            tc.nn.init.zeros_(self._x2fioj.bias)

        self._h2fioj = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=(4 * self._hidden_dim),
            bias=False)
        tc.nn.init.xavier_normal_(self._h2fioj.weight)

        if self._use_ln:
            self._x2fioj_ln = LayerNorm(units=(4 * self._hidden_dim))
            self._h2fioj_ln = LayerNorm(units=(4 * self._hidden_dim))

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

    @property
    def output_dim(self):
        return self._hidden_dim

    def forward(
        self,
        inputs: tc.FloatTensor,
        prev_state: tc.FloatTensor
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Run recurrent state update, compute features.
        Args:
            inputs: input vec tensor with shape [B, ..., ?]
            prev_state: prev lstm state w/ shape [B, 2*H].
        Notes:
            '...' must be either one dimensional or must not exist.
        Returns:
            features, new_state.
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        T = inputs.shape[1]
        features_by_timestep = []
        state = prev_state
        for t in range(0, T):  # 0, ..., T-1
            h_prev, c_prev = tc.chunk(state, 2, dim=-1)
            fioj_from_x = self._x2fioj(inputs[:,t,:])
            fioj_from_h = self._h2fioj(h_prev)
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
            h_new = o * c_new

            features_by_timestep.append(h_new)
            state = tc.cat((h_new, c_new), dim=-1)

        features = tc.stack(features_by_timestep, dim=1)
        if T == 1:
            features = features.squeeze(1)
        new_state = state

        return features, new_state
