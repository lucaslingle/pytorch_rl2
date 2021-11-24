"""
Implements attention operations for RL^2 agents
"""

import functools

import torch as tc


@functools.lru_cache
def sinusoidal_embeddings(src_len, d_model, reverse=False):
    pos_seq = tc.arange(src_len)
    inv_freq = 1 / (10000 ** (tc.arange(0, d_model, 2) / d_model))
    sinusoid_input = pos_seq.view(-1, 1) * inv_freq.view(1, -1)
    pos_emb = tc.cat((tc.sin(sinusoid_input), tc.cos(sinusoid_input)), dim=-1)
    if reverse:
        pos_emb = tc.flip(pos_emb, dims=(0,))
    return pos_emb


@functools.lru_cache
def get_mask(dest_len, src_len):
    i = tc.arange(dest_len).view(dest_len, 1)
    j = tc.arange(src_len).view(1, src_len)
    m = i >= j - (src_len - dest_len)
    return m.int()


def masked_self_attention(qs, ks, vs, use_mask=True):
    scores = tc.bmm(qs, ks.permute(0, 2, 1))
    scores /= qs.shape[-1] ** 0.5

    if use_mask:
        mask = get_mask(dest_len=qs.shape[1], src_len=ks.shape[1])
        mask = mask.view(1, *mask.shape)
        scores = scores * mask - 1e10 * (1 - mask)

    ws = tc.nn.Softmax(dim=-1)(scores)
    output = tc.bmm(ws, vs)
    return output


def rel_shift(inputs):
    # inputs should be a 3d tensor with shape [B, T2, T1+T2]
    # this function implements the part of the shift from Dai et al., Appdx B
    input_shape = inputs.shape
    zp = tc.zeros(size=(input_shape[0], input_shape[1], 1), dtype=tc.float32)
    inputs = tc.cat((zp, inputs), dim=2)
    inputs = tc.reshape(
        inputs, [input_shape[0], input_shape[2]+1, input_shape[1]])
    inputs = inputs[:, 1:, :]
    inputs = tc.reshape(inputs, input_shape)
    return inputs


def relative_masked_self_attention(qs, ks, vs, rs, u_, v_, use_mask=True):
    ac_qs = qs + u_.unsqueeze(1)
    bd_qs = qs + v_.unsqueeze(1)
    ac = tc.bmm(ac_qs, ks.permute(0, 2, 1))
    bd = tc.bmm(bd_qs, rs.permute(0, 2, 1))
    bd = rel_shift(bd)

    bd = bd[:, :, 0:ks.shape[1]]  # this is a no-op unless prev row attn is used
    scores = ac + bd
    scores /= qs.shape[-1] ** 0.5

    if use_mask:
        mask = get_mask(dest_len=qs.shape[1], src_len=ks.shape[1])
        mask = mask.view(1, *mask.shape)
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
            row_len=None
    ):
        assert position_encoding_style in ['abs', 'rel']
        assert attention_style in ['full', 'row', 'previous_row', 'column']
        assert attention_style == 'full' or row_len is not None

        super().__init__()
        self._input_dim = input_dim
        self._num_heads = num_heads
        self._num_head_features = num_head_features
        self._position_encoding_style = position_encoding_style
        self._attention_style = attention_style
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

    def attn_preop(self, qs, ks, vs, sampling):
        assert type(qs) == type(ks) == type(vs)
        assert (sampling and type(qs) == list) or \
               (not sampling and type(qs) == tc.Tensor)

        if self._attention_style == 'full':
            if sampling:
                qs = tc.stack(qs, dim=1)
                ks = tc.stack(ks, dim=1)
                vs = tc.stack(vs, dim=1)
                return qs, ks, vs, qs.shape[0]
            else:
                return qs, ks, vs, qs.shape[0]

        if self._attention_style == 'row':
            if sampling:
                assert len(qs) == 1
                row_idx = (len(ks)-1) // self._row_len
                row_flat_idx = row_idx * self._row_len
                ks = ks[row_flat_idx:]  # get relevant row
                vs = vs[row_flat_idx:]
                qs = tc.stack(qs, dim=1)
                ks = tc.stack(ks, dim=1)
                vs = tc.stack(vs, dim=1)
                return qs, ks, vs, qs.shape[0]
            else:
                assert qs.shape[1] == ks.shape[1] == vs.shape[1]
                assert qs.shape[1] % self._row_len == 0
                qs = tc.reshape(qs, [-1, self._row_len, qs.shape[-1]])
                ks = tc.reshape(ks, [-1, self._row_len, ks.shape[-1]])
                vs = tc.reshape(vs, [-1, self._row_len, vs.shape[-1]])
                return qs, ks, vs, qs.shape[0]

        if self._attention_style == 'previous_row':
            if sampling:
                assert len(qs) == 1
                row_idx = (len(ks)-1) // self._row_len
                if row_idx > 0:
                    prev_row_flat_idx = (row_idx - 1) * self._row_len
                    ks = ks[prev_row_flat_idx:prev_row_flat_idx+self._row_len]
                    vs = vs[prev_row_flat_idx:prev_row_flat_idx+self._row_len]
                    qs = tc.stack(qs, dim=1)
                    ks = tc.stack(ks, dim=1)
                    vs = tc.stack(vs, dim=1)
                    return qs, ks, vs, qs.shape[0]
                else:
                    qs = tc.stack(qs, dim=1)
                    prev_row_shape = [qs.shape[0], self._row_len, qs.shape[2]]
                    ks = tc.zeros(size=prev_row_shape, dtype=tc.float32)
                    vs = tc.zeros(size=prev_row_shape, dtype=tc.float32)
                    return qs, ks, vs, qs.shape[0]
            else:
                assert qs.shape[1] == ks.shape[1] == vs.shape[1]
                assert qs.shape[1] % self._row_len == 0
                n_rows = qs.shape[1] // self._row_len
                qs = tc.reshape(qs, [-1, n_rows, self._row_len, qs.shape[-1]])
                ks = tc.reshape(ks, [-1, n_rows, self._row_len, ks.shape[-1]])
                vs = tc.reshape(vs, [-1, n_rows, self._row_len, vs.shape[-1]])
                ks = tc.nn.functional.pad(ks[:,:-1,:,:], (0,0,0,0,1,0))
                vs = tc.nn.functional.pad(vs[:,:-1,:,:], (0,0,0,0,1,0))
                qs = tc.reshape(qs, [-1, self._row_len, qs.shape[-1]])
                ks = tc.reshape(ks, [-1, self._row_len, ks.shape[-1]])
                vs = tc.reshape(vs, [-1, self._row_len, vs.shape[-1]])
                return qs, ks, vs, qs.shape[0]

        if self._attention_style == 'column':
            if sampling:
                assert len(qs) == 1
                column_flat_idx = (len(ks)-1) % self._row_len
                ks = ks[column_flat_idx::self._row_len]  # get relevant column
                vs = vs[column_flat_idx::self._row_len]
                qs = tc.stack(qs, dim=1)
                ks = tc.stack(ks, dim=1)
                vs = tc.stack(vs, dim=1)
                return qs, ks, vs, qs.shape[0]
            else:
                assert qs.shape[1] == ks.shape[1] == vs.shape[1]
                assert qs.shape[1] % self._row_len == 0
                n_rows = qs.shape[1] // self._row_len
                qs = tc.reshape(qs, [-1, n_rows, self._row_len, qs.shape[-1]])
                ks = tc.reshape(ks, [-1, n_rows, self._row_len, ks.shape[-1]])
                vs = tc.reshape(vs, [-1, n_rows, self._row_len, vs.shape[-1]])
                qs = qs.permute(0, 2, 1, 3)
                ks = ks.permute(0, 2, 1, 3)
                vs = vs.permute(0, 2, 1, 3)
                qs = tc.reshape(qs, [-1, n_rows, qs.shape[-1]])
                ks = tc.reshape(ks, [-1, n_rows, ks.shape[-1]])
                vs = tc.reshape(vs, [-1, n_rows, vs.shape[-1]])
                return qs, ks, vs, qs.shape[0]

        raise NotImplementedError

    def attn_postop(self, attn_out, input_len, sampling):
        if self._attention_style == 'full':
            return attn_out

        assert input_len % self._row_len == 0 or sampling

        if self._attention_style == 'row':
            if sampling:
                return attn_out
            else:
                attn_out = tc.reshape(attn_out, [-1, input_len, attn_out.shape[-1]])
                return attn_out

        if self._attention_style == 'previous_row':
            if sampling:
                return attn_out
            else:
                attn_out = tc.reshape(attn_out, [-1, input_len, attn_out.shape[-1]])
                return attn_out

        if self._attention_style == 'column':
            if sampling:
                return attn_out
            else:
                n_rows = input_len // self._row_len
                transposed_block_shape = [-1, self._row_len, n_rows, attn_out.shape[-1]]
                attn_out = tc.reshape(attn_out, transposed_block_shape)
                attn_out = attn_out.permute(0, 2, 1, 3)
                attn_out = tc.reshape(attn_out, [-1, input_len, attn_out.shape[-1]])
                return attn_out

        raise NotImplementedError

    def split_heads(self, inputs):
        return tc.cat(tc.chunk(inputs, self._num_heads, dim=-1), dim=0)

    def merge_heads(self, inputs):
        return tc.cat(tc.chunk(inputs, self._num_heads, dim=0), dim=-1)

    def forward(self, inputs, past_kvs=None):
        """
        Args:
            inputs: present input tensor with shape [B, T2, I]
            past_kvs: optional past kvs

        Returns:
            output tensor and new kvs
        """
        assert inputs.shape[-1] == self._input_dim
        sampling = (inputs.shape[1] == 1)
        use_mask = (self._attention_style != 'previous_row')

        qkv = self._qkv_linear(inputs)
        qs, ks, vs = tc.chunk(qkv, 3, dim=-1)

        if sampling:
            # unbind for memory-efficient append op
            qs, ks, vs = map(lambda x: [x.squeeze(1)], [qs, ks, vs])
            if past_kvs is not None:
                past_ks, past_vs = past_kvs
                past_ks.extend(ks)
                past_vs.extend(vs)
                ks = past_ks
                vs = past_vs
            new_kvs = (ks, vs)
        else:
            if past_kvs is not None:
                past_ks, past_vs = past_kvs
                ks = tc.cat((past_ks, ks), dim=1)
                vs = tc.cat((past_vs, vs), dim=1)
            new_kvs = (ks, vs)

        qs, ks, vs, bsp = self.attn_preop(qs, ks, vs, sampling)  # [B', ..., H*F]
        qs, ks, vs = map(self.split_heads, [qs, ks, vs])         # [B'*H, ..., F]

        if self._position_encoding_style == 'rel':
            batch_size, src_len, d_model = bsp, ks.shape[1], inputs.shape[-1]
            max_len = src_len
            if self._attention_style == 'previous_row':
                max_len += qs.shape[1]
            r_mat = sinusoidal_embeddings(max_len, d_model, reverse=True)  # [M, I]
            rs = self._r_linear(r_mat)                                     # [M, H*F]

            rs = tc.tile(rs.unsqueeze(0), [batch_size, 1, 1])    # [B', M, H*F]
            u_ = tc.tile(self._u.unsqueeze(0), [batch_size, 1])  # [B', H*F]
            v_ = tc.tile(self._v.unsqueeze(0), [batch_size, 1])  # [B', H*F]
            rs, u_, v_ = map(self.split_heads, [rs, u_, v_])     # [B'*H, ..., F]

            attn_output = relative_masked_self_attention(
                qs, ks, vs, rs, u_, v_, use_mask=use_mask)   # [B'*H, T2', F]
        else:
            attn_output = masked_self_attention(
                qs, ks, vs, use_mask=use_mask)              # [B'*H, T2', F]

        attn_output = self.merge_heads(attn_output)  # [B', T2', H*F]
        attn_output = self.attn_postop(
            attn_output,
            input_len=inputs.shape[1],
            sampling=sampling)  # [B, T2, H*F]

        return attn_output, new_kvs
