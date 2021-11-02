"""
Implements common agent components used in Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

import torch as tc


class LayerNorm(tc.nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, units):
        super().__init__()
        self._units = units
        self._g = tc.nn.Parameter(tc.ones(units, device='cpu'))
        self._b = tc.nn.Parameter(tc.zeros(units, device='cpu'))

    def forward(self, x, eps=1e-8):
        mu = tc.mean(x, dim=-1, keepdim=True)
        zero_mu_x = x - mu
        sigma2 = tc.mean(tc.square(zero_mu_x), dim=-1, keepdim=True)
        sigma = tc.sqrt(sigma2 + eps)
        standardized_x = zero_mu_x / sigma

        g, b = self._g, self._b
        while len(list(g.shape)) < len(list(x.shape)):
            g = g.unsqueeze(0)
            b = b.unsqueeze(0)

        scaled_x = g * standardized_x + b
        return scaled_x


def masked_self_attention(q, k, v):
    # TODO(lucaslingle):
    # implement parameterless version of causal self attention here
    raise NotImplementedError


class MultiheadSelfAttention(tc.nn.Module):
    def __init__(self, input_dim, num_heads, num_head_features, proj_dim):
        super().__init__()
        self._input_dim = input_dim
        self._num_heads = num_heads
        self._num_head_features = num_head_features
        self._proj_dim = proj_dim

        # TODO(lucaslingle):
        # add linear modules and forward method that supports optional memory tensor

    def forward(self, presents, pasts=None):
        raise NotImplementedError


