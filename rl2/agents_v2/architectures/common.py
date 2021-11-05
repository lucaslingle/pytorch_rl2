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

        scaled = g * standardized + b
        return scaled


def get_mask(dest_len, src_len):
    i = tc.arange(dest_len).view(dest_len, 1)
    j = tc.arange(src_len).view(1, src_len)
    m = i >= j - (src_len - dest_len)
    return m.int()


def masked_self_attention(q, k, v):
    mask = get_mask(dest_len=q.shape[1], src_len=k.shape[1])
    mask = mask.view(1, *mask.shape)

    scores = tc.bmm(q, k.permute(0, 2, 1))
    scores /= tc.sqrt(q.shape[-1])
    scores -= 1e10 * (1. - mask)
    w = tc.nn.Softmax(dim=-1)(scores)  # [-1, T2, T1+T2]

    output = tc.bmm(w, v)
    return output


class MultiheadSelfAttention(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            num_heads,
            num_head_features,
            connection_style,
            activation=None
    ):
        assert connection_style in ['plain', 'residual', 'dense']
        super().__init__()
        self._input_dim = input_dim
        self._num_heads = num_heads
        self._num_head_features = num_head_features
        self._connection_style = connection_style
        self._activation = activation

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
        """
        Args:
            inputs: present input tensor with shape [B, T2, I]
            past_kvs: optional past kvs with shape [B, T1, H*F*2]

        Returns:
            output tensor with shape determined by self._connection_style
        """
        qkv = self._qkv_linear(inputs)
        qs, ks, vs = tc.chunk(qkv, 3, dim=-1)

        if past_kvs is not None:
            past_ks, past_vs = tc.chunk(past_kvs, 2, dim=-1)
            ks = tc.cat((past_ks, ks), dim=1)
            vs = tc.cat((past_vs, vs), dim=1)

        new_kvs = tc.cat((ks, vs), dim=-1)  # [B, T1+T2, H*F*2]

        qs, ks, vs = list(map(
            lambda x: tc.cat(tc.chunk(x, self._num_heads, dim=-1), dim=0),
            [qs, ks, vs]))

        attn_output = masked_self_attention(qs, ks, vs)
        attn_output = tc.cat(tc.chunk(attn_output, self._num_heads, dim=0), dim=-1)

        if self._connection_style == 'residual':
            attn_output = self._proj_linear(attn_output)

        if self._activation is not None:
            attn_output = self._activation(attn_output)

        if self._connection_style == 'plain':
            output = attn_output
        elif self._connection_style == 'residual':
            output = inputs + attn_output
        elif self._connection_style == 'dense':
            output = tc.cat((inputs, attn_output), dim=-1)
        else:
            raise NotImplementedError

        return output, new_kvs
