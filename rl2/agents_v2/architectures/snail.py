"""
Implements SNAIL architecture (Mishra et al., 2017) for RL^2.
"""

from typing import Optional

import torch as tc

from rl2.agents_v2.architectures.common import LayerNorm


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
            present_inputs: tc.FloatTensor,
            past_inputs: Optional[tc.FloatTensor] = None
    ) -> tc.FloatTensor:
        """
        Args:
            present_inputs: present input tensor of shape [B, T2, I]
            past_inputs: optional past input tensor of shape [B, T1, I]

        Returns:
            Causal convolution of the (padded) present inputs.
            The present inputs are padded with past inputs (if any),
            and possibly zero padding.
        """
        batch_size = present_inputs.shape[0]
        effective_kernel_size = self.effective_kernel_size()

        if past_inputs is not None:
            t1 = list(past_inputs.shape)[1]
            if t1 < effective_kernel_size - 1:
                zpl = (effective_kernel_size - 1) - t1
                zps = (batch_size, zpl, self._input_dim)
                zp = tc.zeros(size=zps, dtype=tc.float32)
                inputs = tc.cat(
                    (zp, past_inputs, present_inputs),
                    dim=1
                )
            elif t1 > effective_kernel_size - 1:
                inputs = tc.cat(
                    (past_inputs[:, -(effective_kernel_size - 1):], present_inputs),
                    dim=1
                )
            else:
                inputs = tc.cat(
                    (past_inputs, present_inputs),
                    dim=1)
        else:
            zpl = (effective_kernel_size - 1)
            zps = (batch_size, zpl, self._input_dim)
            zp = tc.zeros(size=zps, dtype=tc.float32)
            inputs = tc.cat(
                (zp, present_inputs),
                dim=1
            )

        conv = self._conv(inputs)
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
            feature_dim=2*self._feature_dim,
            kernel_size=self._kernel_size,
            dilation_rate=self._dilation_rate,
            use_bias=(not self._use_ln))

        if self._use_ln:
            self._conv_ln = LayerNorm(units=self._feature_dim)

    def forward(self, present_inputs, past_inputs=None):
        conv = self._conv(
            present_inputs=present_inputs,
            past_inputs=past_inputs)

        if self._use_ln:
            conv = self._conv_ln(conv)

        xg, xf = tc.chunk(conv, 2, dim=-1)
        xg, xf = tc.nn.Sigmoid()(xg), tc.nn.Tanh()(xf)
        activations = xg * xf
        return activations


class TCBlock(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            kernel_size,
            dilation_rate,
            context_size,
            use_ln=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._kernel_size = kernel_size
        self._dilation_rate = dilation_rate
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
        raise NotImplementedError

    def forward(self, input_vec, prev_state):
        if len(list(input_vec.shape)) == 2:
            input_vec = input_vec.unsqueeze(1)

        past_activations = prev_state  # [B, T1, I+(L-1)*F]
        present_activations = input_vec  # [B, T2, I]

        for l in range(0, self.num_layers):  # 0, ..., num_layers-1
            if past_activations is None:
                past_inputs = None
            else:
                end_idx = self._input_dim + l * self._feature_dim
                past_inputs = past_activations[:, :, 0:end_idx]  # [B, T1, I+l*F]

            present_inputs = present_activations  # [B, T2, I+l*F]

            output = self._dense_blocks[l](
                present_inputs=present_inputs,
                past_inputs=past_inputs)  # [B, T2, F]

            present_activations = tc.cat(
                (present_activations, output.unsqueeze(1)),
                dim=1)  # [B, T2, I+(l+1)*F]

        return present_activations  # [B, T2, I+L*F]