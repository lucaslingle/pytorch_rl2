"""
Implements common agent components used in Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

import torch as tc


class Linear(tc.nn.Module):
    """
    A linear layer with optional weight normalization included.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        weight_initializer,
        use_wn=True,
        use_bias=True
    ):
        super().__init__()
        self._use_wn = use_wn
        self._use_bias = use_bias

        weights = tc.empty(output_dim, input_dim, device='cpu')
        weights = weight_initializer(weights)
        self._weights = tc.nn.Parameter(weights)

        if self._use_wn:
            gains = tc.sqrt(tc.sum(tc.square(self._weights.detach()), dim=-1))
            self._gains = tc.nn.Parameter(gains)

        if self._use_bias:
            biases = tc.empty(output_dim, device='cpu')
            biases = tc.nn.init.zeros_(biases)
            self._biases = tc.nn.Parameter(biases)

    def forward(self, x):
        if self._use_wn:
            weight_norms = tc.sqrt(tc.sum(tc.square(self._weights), dim=-1))
            w = self._weights / weight_norms.unsqueeze(-1)
            output = tc.einsum(
                'o,bo->bo', self._gains, tc.einsum('oi,bi->bo', w, x))
        else:
            w = self._weights
            output = tc.einsum('oi,bi->bo', w, x)

        if self._use_bias:
            output += self._biases.unsqueeze(0)

        return output


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


class DuanGRU(tc.nn.Module):
    """
    GRU from Duan et al., 2016.
    """
    def __init__(self, input_dim, hidden_dim, use_wn=True):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._use_wn = use_wn

        self._x2z = Linear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=False)
        self._h2z = Linear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=True)

        self._x2r = Linear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=False)
        self._h2r = Linear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=True)

        self._x2hhat = Linear(
            input_dim=self._input_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.xavier_normal_,
            use_wn=self._use_wn,
            use_bias=False)
        self._rh2hhat = Linear(
            input_dim=self._hidden_dim,
            output_dim=self._hidden_dim,
            weight_initializer=tc.nn.init.orthogonal_,
            use_wn=self._use_wn,
            use_bias=True)

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
    def __init__(self, input_dim, hidden_dim, use_wn=False):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._use_wn = use_wn

        self._linear = Linear(
            input_dim=(self._input_dim + self._hidden_dim),
            output_dim=(4 * self._hidden_dim),
            weight_initializer=lambda m: tc.nn.init.uniform_(m, -0.10, 0.10),
            use_wn=self._use_wn,
            use_bias=True)

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


class PolicyHead(tc.nn.Module):
    """
    Policy head for a reinforcement learning agent.
    Uses a weight-normed linear layer.
    """
    def __init__(self, num_features, num_actions, use_wn=True):
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
    Uses a weight-normed linear layer.
    """
    def __init__(self, num_features, use_wn=True):
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
