"""
Implements SNAIL architecture (Mishra et al., 2017) for RL^2.
"""

from typing import Optional, Tuple

import torch as tc
import numpy as np

from rl2.agents.architectures.common import LayerNorm, MultiheadSelfAttention


class CausalConv(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            kernel_size,
            dilation_rate,
            use_bias=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate
        self._use_bias = use_bias

        self._conv = tc.nn.Conv1d(
            in_channels=self._input_dim,
            out_channels=self._feature_dim,
            kernel_size=self._kernel_size,
            stride=(1,),
            padding=(0,),
            dilation=self._dilation_rate,
            bias=self._use_bias)

    @property
    def effective_kernel_size(self):
        k = self._kernel_size
        r = self._dilation_rate
        return k + (k - 1) * (r - 1)

    def forward(
            self,
            inputs: tc.FloatTensor,
            past_inputs: Optional[tc.FloatTensor] = None
    ) -> tc.FloatTensor:
        """
        Args:
            inputs: present input tensor of shape [B, T2, I]
            past_inputs: optional past input tensor of shape [B, T1, I]

        Returns:
            Causal convolution of the (padded) present inputs.
            The present inputs are padded with past inputs (if any),
            and possibly zero padding.
        """
        batch_size = inputs.shape[0]
        effective_kernel_size = self.effective_kernel_size

        if past_inputs is not None:
            t1 = list(past_inputs.shape)[1]
            if t1 < effective_kernel_size - 1:
                zpl = (effective_kernel_size - 1) - t1
                zps = (batch_size, zpl, self._input_dim)
                zp = tc.zeros(size=zps, dtype=tc.float32)
                inputs = tc.cat((zp, past_inputs, inputs), dim=1)
            elif t1 > effective_kernel_size - 1:
                inputs = tc.cat(
                    (past_inputs[:, -(effective_kernel_size - 1):], inputs),
                    dim=1
                )
            else:
                inputs = tc.cat((past_inputs, inputs), dim=1)
        else:
            zpl = (effective_kernel_size - 1)
            zps = (batch_size, zpl, self._input_dim)
            zp = tc.zeros(size=zps, dtype=tc.float32)
            inputs = tc.cat((zp, inputs), dim=1)

        conv = self._conv(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        return conv


class DenseBlock(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            kernel_size,
            dilation_rate,
            use_ln=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate
        self._use_ln = use_ln

        self._conv = CausalConv(
            input_dim=self._input_dim,
            feature_dim=(2 * self._feature_dim),
            kernel_size=self._kernel_size,
            dilation_rate=self._dilation_rate,
            use_bias=(not self._use_ln))

        if self._use_ln:
            self._conv_ln = LayerNorm(units=(2 * self._feature_dim))

    def forward(self, inputs, past_inputs=None):
        conv = self._conv(inputs=inputs, past_inputs=past_inputs)

        if self._use_ln:
            conv = self._conv_ln(conv)

        xg, xf = tc.chunk(conv, 2, dim=-1)
        xg, xf = tc.nn.Sigmoid()(xg), tc.nn.Tanh()(xf)
        activations = xg * xf

        output = tc.cat((inputs, activations), dim=-1)
        return output


class TCBlock(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            context_size,
            use_ln=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._context_size = context_size
        self._use_ln = use_ln

        self._dense_blocks = tc.nn.ModuleList([
            DenseBlock(
                input_dim=(self._input_dim + l * self._feature_dim),
                feature_dim=self._feature_dim,
                kernel_size=2,
                dilation_rate=2 ** l,
                use_ln=self._use_ln)
            for l in range(0, self.num_layers)
        ])

    @property
    def num_layers(self):
        log2_context_size = np.log(self._context_size) / np.log(2)
        return int(np.ceil(log2_context_size))

    def forward(self, inputs, past_inputs=None):
        """
        Args:
            inputs: inputs tensor of shape [B, T2, I]
            past_inputs: optional past inputs tensor of shape [B, T1, I+L*F]

        Returns:
            tensor of shape [B, T2, I+L*F]
        """
        for l in range(0, self.num_layers):  # 0, ..., num_layers-1
            if past_inputs is None:
                past_inputs_for_layer = None
            else:
                end_idx = self._input_dim + l * self._feature_dim
                past_inputs_for_layer = past_inputs[:, :, 0:end_idx]

            inputs = self._dense_blocks[l](
                inputs=inputs, past_inputs=past_inputs_for_layer)

        return inputs  # [B, T2, I+L*F]


class SNAIL(tc.nn.Module):
    def __init__(self, input_dim, feature_dim, context_size, use_ln=True):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._context_size = context_size
        self._use_ln = use_ln

        self._tc1 = TCBlock(
            input_dim=self._input_dim,
            feature_dim=self._feature_dim,
            context_size=self._context_size,
            use_ln=self._use_ln)

        self._tc1_output_dim = self._input_dim + \
                               self._tc1.num_layers * self._feature_dim
        self._tc2 = TCBlock(
            input_dim=self._tc1_output_dim,
            feature_dim=self._feature_dim,
            context_size=self._context_size,
            use_ln=self._use_ln)

        self._tc2_output_dim = self._input_dim + \
                               self._tc1.num_layers * self._feature_dim + \
                               self._tc2.num_layers * self._feature_dim
        self._attn = MultiheadSelfAttention(
            input_dim=self._tc2_output_dim,
            num_heads=1,
            num_head_features=self._feature_dim,
            position_encoding_style='abs',
            attention_style='full',
            connection_style='dense',
            use_ln=False)

    def initial_state(self, batch_size: int) -> None:
        return None

    @property
    def output_dim(self):
        return self._tc2_output_dim + self._feature_dim

    def forward(
        self,
        inputs: tc.FloatTensor,
        prev_state: Optional[tc.FloatTensor]
    ) -> Tuple[tc.FloatTensor, tc.FloatTensor]:
        """
        Run state update, compute features.

        Args:
            inputs: input vec tensor with shape [B, ..., ?]
            prev_state: previous architecture state

        Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            tuple containing features with shape [B, ..., F], and new_state.
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        if prev_state is None:
            tc1_out = self._tc1(inputs=inputs, past_inputs=None)
            tc2_out = self._tc2(inputs=tc1_out, past_inputs=None)
            attn_out, new_attn_kv = self._attn(inputs=tc2_out, past_kvs=None)

            # TODO(lucaslingle):
            #  ugly, all due to overhauling attn code;
            #  added this for compatibility.
            #  fix later by adding support for tensor-typed new_kvs
            #  back to multiheadselfattention, make it optional this time.
            new_ks, new_vs = new_attn_kv
            new_ks, new_vs = tc.stack(new_ks, dim=1), tc.stack(new_vs, dim=1)
            new_attn_kv = tc.cat((new_ks, new_vs), dim=-1)

            features = attn_out
            new_state = tc.cat((tc2_out, new_attn_kv), dim=-1)

            if features.shape[1] == 1:
                features = features.squeeze(1)

            return features, new_state

        tc1_out = self._tc1(
            inputs=inputs, past_inputs=prev_state[:, :, 0:self._tc1_output_dim])

        tc2_out = self._tc2(
            inputs=tc1_out, past_inputs=prev_state[:, :, 0:self._tc2_output_dim])

        # TODO(lucaslingle): see above todo
        past_kvs = prev_state[:, :, self._tc2_output_dim:]
        past_ks, past_vs = tc.chunk(past_kvs, 2, dim=-1)
        past_ks, past_vs = list(map(
            lambda x: list(tc.unbind(x, dim=1)), [past_ks, past_vs]))
        past_kvs = (past_ks, past_vs)

        attn_out, new_attn_kv = self._attn(
            inputs=tc2_out, past_kvs=past_kvs)

        # TODO(lucaslingle): see above todo
        new_ks, new_vs = new_attn_kv
        new_ks, new_vs = tc.stack(new_ks, dim=1), tc.stack(new_vs, dim=1)
        new_attn_kv = tc.cat((new_ks, new_vs), dim=-1)
        new_attn_kv = new_attn_kv[:, -tc2_out.shape[1]:, :]

        features = attn_out
        new_state = tc.cat(
            (prev_state, tc.cat((tc2_out, new_attn_kv), dim=-1)),
            dim=1)

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_state
