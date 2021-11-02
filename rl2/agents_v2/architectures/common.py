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

    def forward(self, inputs, eps=1e-8):
        mu = tc.mean(inputs, dim=-1, keepdim=True)
        centered = inputs - mu
        sigma2 = tc.mean(tc.square(centered), dim=-1, keepdim=True)
        sigma = tc.sqrt(sigma2 + eps)
        standardized = centered / sigma

        g, b = self._g, self._b
        while len(list(g.shape)) < len(list(inputs.shape)):
            g = g.unsqueeze(0)
            b = b.unsqueeze(0)

        scaled_x = g * standardized + b
        return scaled_x


def masked_self_attention(q, k, v):
    # TODO(lucaslingle):
    # implement parameterless version of causal self attention here
    raise NotImplementedError


class MultiheadSelfAttention(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            num_heads,
            num_head_features,
            connection_style
    ):
        assert connection_style in ['plain', 'residual', 'dense']
        super().__init__()
        self._input_dim = input_dim
        self._num_heads = num_heads
        self._num_head_features = num_head_features
        self._connection_style = connection_style

        # TODO(lucaslingle):
        # add linear modules and forward method that supports optional memory tensor

        self._qkv_linear = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=(self._num_heads * self._num_head_features * 3),
            bias=False)
        tc.nn.init.xavier_normal_(self._qkv_linear.weight)

        if self._connection_style == 'residual':
            self._proj_linear = tc.nn.Linear(
                in_features=(self._num_heads * self._num_head_features),
                out_features=input_dim,
                bias=False)
            tc.nn.init.xavier_normal_(self._proj_linear.weight)

    def forward(self, inputs, past_kvs=None):
        qkv = self._qkv_linear(inputs)
        qkv_list = tc.chunk(qkv, 3, dim=-1)
        qs, ks, vs = list(map(
            lambda x: tc.cat(tc.chunk(x, self._num_heads, dim=-1), dim=0),
            qkv_list))

        if past_kvs is not None:
            past_ks, past_vs = tc.chunk(past_kvs, 2, dim=-1)
            ks = tc.cat((past_ks, ks), dim=1)
            vs = tc.cat((past_vs, vs), dim=1)

        attn_output = masked_self_attention(qs, ks, vs)
        attn_output = tc.cat(tc.chunk(attn_output, self._num_heads, 0), dim=-1)

        if self._connection_style == 'plain':
            return attn_output
        elif self._connection_style == 'residual':
            return inputs + self._proj_linear(attn_output)
        elif self._connection_style == 'dense':
            return tc.cat((inputs, attn_output), dim=-1)
        else:
            raise NotImplementedError


