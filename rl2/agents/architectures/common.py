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


def sinusoidal_embeddings(src_len, d_model):
    pos_seq = tc.arange(src_len)
    inv_freq = 1 / (10000 ** (tc.arange(0, d_model, 2) / d_model))
    sinusoid_input = pos_seq.view(-1, 1) * inv_freq.view(1, -1)
    pos_emb = tc.cat((tc.sin(sinusoid_input), tc.cos(sinusoid_input)), dim=-1)
    return pos_emb


def get_mask(dest_len, src_len):
    i = tc.arange(dest_len).view(dest_len, 1)
    j = tc.arange(src_len).view(1, src_len)
    m = i >= j - (src_len - dest_len)
    return m.int()


def masked_self_attention(q, k, v):
    mask = get_mask(dest_len=q.shape[1], src_len=k.shape[1])
    mask = mask.view(1, *mask.shape)

    scores = tc.bmm(q, k.permute(0, 2, 1))
    scores /= q.shape[-1] ** 0.5
    scores -= 1e10 * (1 - mask)
    w = tc.nn.Softmax(dim=-1)(scores)  # [-1, T2, T1+T2]

    output = tc.bmm(w, v)
    return output


def rel_shift(inputs):
    # inputs should be a 3d tensor with shape [B, T2, T1+T2]
    # this function implements the part of the shift from Dai et al., Appdx B,
    # but must be combined with subsequent causal masking to have correct effect
    input_shape = inputs.shape
    zp = tc.zeros(size=(input_shape[0], input_shape[1], 1), dtype=tc.float32)
    inputs = tc.cat((zp, inputs), dim=2)
    inputs = tc.reshape(
        inputs, [input_shape[0], input_shape[2]+1, input_shape[1]])
    inputs = inputs[:, 1:, :]
    inputs = tc.reshape(inputs, input_shape)
    return inputs


def relative_masked_self_attention(qs, ks, vs, rs, u_, v_):
    mask = get_mask(dest_len=qs.shape[1], src_len=ks.shape[1])
    mask = mask.view(1, *mask.shape)

    ac_qs = qs + u_.unsqueeze(1)
    bd_qs = qs + v_.unsqueeze(1)
    ac = tc.bmm(ac_qs, ks.permute(0, 2, 1))
    bd = tc.bmm(bd_qs, rs.permute(0, 2, 1))
    bd = rel_shift(bd)

    scores = ac + bd
    scores /= qs.shape[-1] ** 0.5
    scores = scores * mask - 1e10 * (1 - mask)
    ws = tc.nn.Softmax(dim=-1)(scores)

    output = tc.bmm(ws, vs)
    return output


class MultiheadSelfAttention(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            num_heads,
            num_head_features,
            position_encoding_style,
            attention_style,
            connection_style,
            activation=None,
            use_ln=True,
            row_len=None
    ):
        assert position_encoding_style in ['abs', 'rel']
        assert attention_style in ['full', 'locally_banded_dense', 'strided_sparse']
        assert connection_style in ['plain', 'residual', 'dense']
        assert attention_style == 'full' or row_len is not None

        super().__init__()
        self._input_dim = input_dim
        self._num_heads = num_heads
        self._num_head_features = num_head_features
        self._position_encoding_style = position_encoding_style
        self._attention_style = attention_style
        self._connection_style = connection_style
        self._activation = activation
        self._use_ln = use_ln
        self._row_len = row_len

        self._qkv_linear = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=(self._num_heads * self._num_head_features * 3),
            bias=False)
        tc.nn.init.xavier_normal_(self._qkv_linear.weight)

        if self._position_encoding_style == 'rel':
            self._r_linear = tc.nn.Linear(
                in_features=self._input_dim,
                out_features=(self._num_heads * self._num_head_features),
                bias=False)
            tc.nn.init.xavier_normal_(self._r_linear.weight)
            self._u = tc.nn.Parameter(
                tc.zeros(size=(self._num_heads * self._num_head_features,),
                         dtype=tc.float32))
            self._v = tc.nn.Parameter(
                tc.zeros(size=(self._num_heads * self._num_head_features,),
                         dtype=tc.float32))

        if self._connection_style == 'residual':
            self._proj_linear = tc.nn.Linear(
                in_features=(self._num_heads * self._num_head_features),
                out_features=input_dim,
                bias=False)
            tc.nn.init.xavier_normal_(self._proj_linear.weight)

    def attn_preop(self, qs, ks, vs, sampling):
        if self._attention_style == 'full':
            return qs, ks, vs, qs.shape[0]

        if self._attention_style == 'locally_banded_dense':
            if sampling:
                assert qs.shape[1] == 1
                mod = ks.shape[1] % self._row_len
                ks = ks[:, -mod:, :]  # get relevant row
                vs = vs[:, -mod:, :]
                return qs, ks, vs, qs.shape[0]
            else:
                assert qs.shape[1] == ks.shape[1] == vs.shape[1]
                assert qs.shape[1] % self._row_len == 0
                qs = tc.reshape(qs, [-1, self._row_len, qs.shape[-1]])
                ks = tc.reshape(ks, [-1, self._row_len, ks.shape[-1]])
                vs = tc.reshape(vs, [-1, self._row_len, vs.shape[-1]])
                return qs, ks, vs, qs.shape[0]

        elif self._attention_style == 'strided_sparse':
            if sampling:
                assert qs.shape[1] == 1
                mod = ks.shape[1] % self._row_len
                ks = ks[:, (mod-1)::self._row_len, :]  # get relevant column
                vs = vs[:, (mod-1)::self._row_len, :]
                return qs, ks, vs, qs.shape[0]
            else:
                assert qs.shape[1] == ks.shape[1] == vs.shape[1]
                assert qs.shape[1] % self._row_len == 0

                batch_size = qs.shape[0]
                n_rows = qs.shape[1] // self._row_len
                n_feature = qs.shape[2]
                block_shape = [batch_size, n_rows, self._row_len, n_feature]

                qs = tc.reshape(qs, block_shape)
                ks = tc.reshape(ks, block_shape)
                vs = tc.reshape(vs, block_shape)

                qs = qs.permute(0, 2, 1, 3)
                ks = ks.permute(0, 2, 1, 3)
                vs = vs.permute(0, 2, 1, 3)

                qs = tc.reshape(qs, [-1, n_rows, n_feature])
                ks = tc.reshape(ks, [-1, n_rows, n_feature])
                vs = tc.reshape(vs, [-1, n_rows, n_feature])
                return qs, ks, vs, qs.shape[0]

        else:
            raise NotImplementedError

    def attn_postop(self, attn_out, input_len, sampling):
        assert input_len % self._row_len == 0 or sampling

        if self._attention_style == 'full':
            return attn_out

        if self._attention_style == 'locally_banded_dense':
            return attn_out

        if self._attention_style == 'strided_sparse':
            if sampling:
                return attn_out
            else:
                n_rows = input_len // self._row_len
                transposed_block_shape = [-1, self._row_len, n_rows, attn_out.shape[-1]]
                attn_out = tc.reshape(attn_out, transposed_block_shape)
                attn_out = attn_out.permute(0, 2, 1, 3)
                attn_out = tc.reshape(attn_out, [-1, input_len, attn_out.shape[-1]])
                return attn_out

    def split_heads(self, inputs):
        return tc.cat(tc.chunk(inputs, self._num_heads, dim=-1), dim=0)

    def merge_heads(self, inputs):
        return tc.cat(tc.chunk(inputs, self._num_heads, dim=0), dim=-1)

    def forward(self, inputs, past_kvs=None):
        """
        Args:
            inputs: present input tensor with shape [B, T2, I]
            past_kvs: optional past kvs with shape [B, T1, H*F*2]

        Returns:
            output tensor with shape determined by self._connection_style
            and new_kvs tensor of shape [B, T1+T2, H*F*2]
        """
        assert inputs.shape[-1] == self._input_dim
        sampling = (inputs.shape[1] == 1)

        qkv = self._qkv_linear(inputs)
        qs, ks, vs = tc.chunk(qkv, 3, dim=-1)

        if past_kvs is not None:
            past_ks, past_vs = tc.chunk(past_kvs, 2, dim=-1)
            ks = tc.cat((past_ks, ks), dim=1)
            vs = tc.cat((past_vs, vs), dim=1)
        new_kvs = tc.cat((ks, vs), dim=-1)  # [B, T1+T2, H*F*2]

        qs, ks, vs, bsp = self.attn_preop(qs, ks, vs, sampling)  # [B', ..., H*F]
        qs, ks, vs = list(map(self.split_heads, [qs, ks, vs]))   # [B'*H, ..., F]

        if self._position_encoding_style == 'rel':
            batch_size, src_len, d_model = bsp, ks.shape[1], inputs.shape[-1]
            r_mat = tc.flip(
                sinusoidal_embeddings(src_len, d_model), dims=(0,))    # [(T1+T2)', I]
            rs = self._r_linear(r_mat)                                 # [(T1+T2)', H*F]

            rs = tc.tile(rs.unsqueeze(0), [batch_size, 1, 1])    # [B', (T1+T2)', H*F]
            u_ = tc.tile(self._u.unsqueeze(0), [batch_size, 1])  # [B', H*F]
            v_ = tc.tile(self._v.unsqueeze(0), [batch_size, 1])  # [B', H*F]

            rs, u_, v_ = list(map(self.split_heads, [rs, u_, v_]))  # [B'*H, ..., F]

            attn_output = relative_masked_self_attention(
                qs, ks, vs, rs, u_, v_)   # [B'*H, T2, F]
        else:
            attn_output = masked_self_attention(qs, ks, vs)   # [B'*H, T2', F]

        attn_output = self.merge_heads(attn_output)  # [B', T2', H*F]
        attn_output = self.attn_postop(
            attn_output,
            input_len=inputs.shape[1],
            sampling=sampling)  # [B, T2, H*F]

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
