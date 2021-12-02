"""
Implements GRU for RL^2.
"""

from typing import Tuple

import torch as tc

from rl2.agents.architectures.common.normalization import LayerNorm


class GRU(tc.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        forget_bias=1.0,
        use_ln=True,
        reset_after=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._forget_bias = forget_bias
        self._use_ln = use_ln
        self._reset_after = reset_after

        self._x2zr = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=(2 * self._hidden_dim),
            bias=(not self._use_ln))
        tc.nn.init.xavier_normal_(self._x2zr.weight)
        if not self._use_ln:
            tc.nn.init.zeros_(self._x2zr.bias)

        self._h2zr = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=(2 * self._hidden_dim),
            bias=False)
        tc.nn.init.xavier_normal_(self._h2zr.weight)

        self._x2hhat = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=self._hidden_dim,
            bias=(not self._use_ln))
        tc.nn.init.xavier_normal_(self._x2hhat.weight)
        if not self._use_ln:
            tc.nn.init.zeros_(self._x2hhat.bias)

        self._h2hhat = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._hidden_dim,
            bias=False)
        tc.nn.init.orthogonal_(self._h2hhat.weight)

        if self._use_ln:
            self._x2zr_ln = LayerNorm(units=(2 * self._hidden_dim))
            self._h2zr_ln = LayerNorm(units=(2 * self._hidden_dim))
            self._x2hhat_ln = LayerNorm(units=self._hidden_dim)
            self._h2hhat_ln = LayerNorm(units=self._hidden_dim)

        self._initial_state = tc.zeros(self._hidden_dim)

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
            prev_state: prev hidden state w/ shape [B, H].
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
            h_prev = state
            zr_from_x = self._x2zr(inputs[:,t,:])
            zr_from_h = self._h2zr(h_prev)
            if self._use_ln:
                zr_from_x = self._x2zr_ln(zr_from_x)
                zr_from_h = self._h2zr_ln(zr_from_h)
            zr = zr_from_x + zr_from_h
            z, r = tc.chunk(zr, 2, dim=-1)

            z = tc.nn.Sigmoid()(z-self._forget_bias)
            r = tc.nn.Sigmoid()(r)

            if self._reset_after:
                hhat_from_x = self._x2hhat(inputs[:,t,:])
                hhat_from_h = self._h2hhat(h_prev)
                if self._use_ln:
                    hhat_from_x = self._x2hhat_ln(hhat_from_x)
                    hhat_from_h = self._h2hhat_ln(hhat_from_h)
                hhat = hhat_from_x + r * hhat_from_h
            else:
                hhat_from_x = self._x2hhat(inputs[:,t,:])
                hhat_from_h = self._h2hhat(r * h_prev)
                if self._use_ln:
                    hhat_from_x = self._x2hhat_ln(hhat_from_x)
                    hhat_from_h = self._h2hhat_ln(hhat_from_h)
                hhat = hhat_from_x + hhat_from_h

            hhat = tc.nn.ReLU()(hhat)
            h_new = (1. - z) * h_prev + z * hhat

            features_by_timestep.append(h_new)
            state = h_new

        features = tc.stack(features_by_timestep, dim=1)
        if T == 1:
            features = features.squeeze(1)
        new_state = state

        return features, new_state
