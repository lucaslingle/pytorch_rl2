"""
Implements common agent components used in Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

import torch as tc


class LayerNorm(tc.nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, units):
        super().__init__()
        gains = tc.empty(units, device='cpu')
        gains = tc.nn.init.ones_(gains)
        self._ln_gains = tc.nn.Parameter(gains, requires_grad=True)

        biases = tc.empty(units, device='cpu')
        biases = tc.nn.init.zeros_(biases)
        self._ln_biases = tc.nn.Parameter(biases, requires_grad=True)

    def forward(self, x):
        mu = tc.mean(x, dim=-1, keepdim=True)
        zero_mu_x = x - mu
        sigma2 = tc.mean(tc.square(zero_mu_x), dim=-1, keepdim=True)
        sigma = tc.sqrt(sigma2 + 1e-8)
        standardized_x = zero_mu_x / sigma
        g, b = self._ln_gains.unsqueeze(0), self._ln_biases.unsqueeze(0)
        scaled_x = g * standardized_x + b
        return scaled_x


class Linear(tc.nn.Module):
    """
    A linear layer with support for weight normalization.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        weight_initializer,
        use_wn=False,
        use_bias=False
    ):
        super().__init__()
        self._use_wn = use_wn
        self._use_bias = use_bias

        weights = tc.empty(output_dim, input_dim, device='cpu')
        weights = weight_initializer(weights)
        self._weights = tc.nn.Parameter(weights, requires_grad=True)

        if self._use_wn:
            gains = tc.sqrt(tc.sum(tc.square(self._weights.detach()), dim=-1))
            self._wn_gains = tc.nn.Parameter(gains, requires_grad=True)

        if self._use_bias:
            biases = tc.empty(output_dim, device='cpu')
            biases = tc.nn.init.zeros_(biases)
            self._biases = tc.nn.Parameter(biases, requires_grad=True)

    def forward(self, x):
        if self._use_wn:
            weight_norms = tc.sqrt(tc.sum(tc.square(self._weights), dim=-1))
            w = self._weights / weight_norms.unsqueeze(-1)
            output = tc.einsum(
                'o,bo->bo', self._wn_gains, tc.einsum('oi,bi->bo', w, x))
        else:
            w = self._weights
            output = tc.einsum('oi,bi->bo', w, x)

        if self._use_bias:
            output += self._biases.unsqueeze(0)

        return output


class DuanGRU(tc.nn.Module):
    """
    GRU from Duan et al., 2016.
    """
    def __init__(
            self, input_dim, hidden_dim, use_wn=False, use_ln=True,
            forget_bias=1.0, reset_after=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._use_wn = use_wn
        self._use_ln = use_ln
        self._forget_bias = forget_bias
        self._reset_after = reset_after

        self._x2zr = Linear(
            input_dim=self._input_dim,
            output_dim=(2 * self._hidden_dim),
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=(not self._use_ln))
        self._h2zr = Linear(
            input_dim=self._hidden_dim,
            output_dim=(2 * self._hidden_dim),
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=False)

        self._x2hhat = Linear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=(not self._use_ln))
        self._h2hhat = Linear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.orthogonal_,
            use_wn=self._use_wn,
            use_bias=False)

        if self._use_ln:
            self._x2zr_ln = LayerNorm(units=(2 * self._hidden_dim))
            self._h2zr_ln = LayerNorm(units=(2 * self._hidden_dim))
            self._x2hhat_ln = LayerNorm(units=self._hidden_dim)
            self._h2hhat_ln = LayerNorm(units=self._hidden_dim)

    def forward(
        self,
        input_vec: tc.LongTensor,
        prev_state: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Run recurrent state update.
        Args:
            input_vec: current timestep input vector as tc.FloatTensor
            prev_state: prev hidden state w/ shape [B, H].

        Returns:
            new_state.
        """
        zr_from_x = self._x2zr(input_vec)
        zr_from_h = self._h2zr(prev_state)
        if self._use_ln:
            zr_from_x = self._x2zr_ln(zr_from_x)
            zr_from_h = self._h2zr_ln(zr_from_h)
        zr = zr_from_x + zr_from_h
        z, r = tc.chunk(zr, 2, dim=-1)

        z = tc.nn.Sigmoid()(z-self._forget_bias)
        r = tc.nn.Sigmoid()(r)

        if self._reset_after:
            hhat_from_x = self._x2hhat(input_vec)
            hhat_from_h = self._h2hhat(prev_state)
            if self._use_ln:
                hhat_from_x = self._x2hhat_ln(hhat_from_x)
                hhat_from_h = self._h2hhat_ln(hhat_from_h)
            hhat = hhat_from_x + r * hhat_from_h
        else:
            hhat_from_x = self._x2hhat(input_vec)
            hhat_from_h = self._h2hhat(r * prev_state)
            if self._use_ln:
                hhat_from_x = self._x2hhat_ln(hhat_from_x)
                hhat_from_h = self._h2hhat_ln(hhat_from_h)
            hhat = hhat_from_x + hhat_from_h

        hhat = tc.nn.ReLU()(hhat)

        h_new = (1. - z) * prev_state + z * hhat
        return h_new


class LSTM(tc.nn.Module):
    """
    LSTM.
    """
    def __init__(
            self, input_dim, hidden_dim, use_wn=False, use_ln=True,
            forget_bias=1.0
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._use_wn = use_wn
        self._use_ln = use_ln
        self._forget_bias = forget_bias

        self._x2fioj = Linear(
            input_dim=self._input_dim,
            output_dim=(4 * self._hidden_dim),
            weight_initializer=lambda m: tc.nn.init.uniform_(m, -0.10, 0.10),
            use_wn=self._use_wn,
            use_bias=(not self._use_ln))

        self._h2fioj = Linear(
            input_dim=self._hidden_dim,
            output_dim=(4 * self._hidden_dim),
            weight_initializer=lambda m: tc.nn.init.uniform_(m, -0.10, 0.10),
            use_wn=self._use_wn,
            use_bias=(not self._use_ln))

        if self._use_ln:
            self._x2fioj_ln = LayerNorm(units=(4 * self._hidden_dim))
            self._h2fioj_ln = LayerNorm(units=(4 * self._hidden_dim))
            self._c_out_ln = LayerNorm(units=self._hidden_dim)

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
        return h_new, c_new


def lstm_postprocessing(
        hidden_state: tc.FloatTensor,
        cell_state: tc.FloatTensor
    ) -> tc.FloatTensor:
        state = tc.cat((hidden_state, cell_state), dim=-1).float()
        return state


class PolicyHead(tc.nn.Module):
    """
    Policy head for a reinforcement learning agent.
    """
    def __init__(self, num_features, num_actions, use_wn=False):
        super().__init__()
        self._num_features = num_features
        self._num_actions = num_actions
        self._use_wn = use_wn
        self._linear = Linear(
            input_dim=self._num_features,
            output_dim=self._num_actions,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=True)

    def forward(self, features: tc.FloatTensor) -> tc.distributions.Categorical:
        """
        Computes a policy distributions from features and returns it.
        Args:
            features: a tc.FloatTensor of features of shape [B, num_features].
        Returns:
            a tc.distributions.Categorical over actions, with batch shape [B].
        """
        logits = self._linear(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class ValueHead(tc.nn.Module):
    """
    Value head for a reinforcement learning agent.
    """
    def __init__(self, num_features, use_wn=False):
        super().__init__()
        self._num_features = num_features
        self._use_wn = use_wn
        self._linear = Linear(
            input_dim=self._num_features,
            output_dim=1,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=True)

    def forward(self, features: tc.FloatTensor) -> tc.FloatTensor:
        """
        Computes a value estimate from features and returns it.
        Args:
            features: a tc.FloatTensor of features with shape [B, num_features].
        Returns:
            a tc.FloatTensor of value estimates with shape [B].
        """
        v_pred = self._linear(features).squeeze(-1)
        return v_pred


def one_hot(ys: tc.LongTensor, depth: int) -> tc.FloatTensor:
    """
    Applies one-hot encoding to a batch of vectors.

    Args:
        ys: tc.LongTensor of shape [B].
        depth: int specifying the number of possible y values.

    Returns:
        the one-hot encodings of tensor ys.
    """

    batch_size = ys.shape[0]
    vecs_shape = [batch_size, depth]
    vecs = tc.zeros(dtype=tc.float32, size=vecs_shape)
    vecs.scatter_(dim=1, index=ys.unsqueeze(-1),
                  src=tc.ones(dtype=tc.float32, size=vecs_shape))
    return vecs.float()
