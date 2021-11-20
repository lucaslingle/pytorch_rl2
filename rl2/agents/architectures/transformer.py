"""
Implements Transformer architectures for RL^2.
"""

from typing import Optional
import math

import torch as tc

from rl2.agents.architectures.common import LayerNorm, MultiheadSelfAttention


class FF(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self._lin1 = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=self._hidden_dim,
            bias=True)
        tc.nn.init.xavier_normal_(self._lin1.weight)
        tc.nn.init.zeros_(self._lin1.bias)

        self._lin2 = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._output_dim,
            bias=True)
        tc.nn.init.zeros_(self._lin2.weight)
        tc.nn.init.zeros_(self._lin2.bias)

    def forward(self, inputs):
        """
        Args:
            inputs: input vec tensor of shape [B, T2, I]

        Returns:
            output tensor of shape [B, T2, O]
        """
        x = inputs
        x = self._lin1(x)
        x = tc.nn.ReLU()(x)
        x = self._lin2(x)
        return x


class TransformerLayer(tc.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            d_head: int,
            d_ff: int,
            position_encoding_style: str,
            attention_style: str,
            layer_ordering: str,
            row_len: Optional[int] = None
    ):
        """
        Args:
            d_model: dimensionality of model features.
            n_head: number of attention heads
            d_head: dimensionality of attention head.
            d_ff: dimensionality of inner layer of feedforward neural network.
            position_encoding_style: one of 'abs', 'rel'.
            attention_style: one of 'full', 'row', 'column', or 'previous_row'.
            layer_ordering: ordering of activation, function, and normalization.
                should be the letters chosen from 'a', 'f', 'n' in any order.
                letter 'f' cannot be omitted.
            row_len: required if attention_style is not 'full'
        """
        assert attention_style == 'full' or row_len is not None
        letters = set(list(layer_ordering))
        assert len(letters) <= 3
        assert 'f' in letters
        assert letters <= {'a', 'f', 'n'}

        super().__init__()
        self._d_model = d_model
        self._n_head = n_head
        self._d_head = d_head
        self._d_ff = d_ff
        self._position_encoding_style = position_encoding_style
        self._attention_style = attention_style
        self._layer_ordering = list(layer_ordering)
        self._row_len = row_len

        if 'a' in self._layer_ordering:
            self._attn_act = tc.nn.ReLU()
            self._ff_act = tc.nn.ReLU()

        if 'n' in self._layer_ordering:
            self._attn_layer_norm = LayerNorm(units=self._d_model)
            self._ff_layer_norm = LayerNorm(units=self._d_model)

        self._attn = MultiheadSelfAttention(
            input_dim=self._d_model,
            num_heads=self._n_head,
            num_head_features=self._d_head,
            position_encoding_style=self._position_encoding_style,
            attention_style=self._attention_style,
            row_len=self._row_len)

        self._proj = tc.nn.Linear(
            in_features=(self._n_head * self._d_head),
            out_features=self._d_model,
            bias=False)
        tc.nn.init.zeros_(self._proj.weight)

        self._ff = FF(
            input_dim=self._d_model,
            hidden_dim=self._d_ff,
            output_dim=self._d_model)

    def forward(self, inputs, past_kvs=None):
        """
        Args:
            inputs: input vec tensor of shape [B, T2, I]
            past_kvs: optional past kvs with shape [B, T1, H*F*2]

        Returns:
            output tensor of shape [B, T2, I]
            and new_kvs tensor of shape [B, T1+T2, H*F*2]
        """
        x = inputs

        i = inputs
        for letter in self._layer_ordering:
            if letter == 'a':
                x = self._attn_act(x)
            elif letter == 'n':
                x = self._attn_layer_norm(x)
            elif letter == 'f':
                attn_output, new_kvs = self._attn(x, past_kvs=past_kvs)
                attn_output = self._proj(attn_output)
                x = i + attn_output
            else:
                raise NotImplementedError

        i = x
        for letter in self._layer_ordering:
            if letter == 'a':
                x = self._ff_act(x)
            elif letter == 'n':
                x = self._ff_layer_norm(x)
            elif letter == 'f':
                ff_output = self._ff(x)
                x = i + ff_output
            else:
                raise NotImplementedError

        return x, new_kvs


class TransformerXLI(tc.nn.Module):
    """
    Implements a variant of Transformer XL-I from Parisotto et al., 2019.

    Note we place the relu after the residual sum, which allows positive coordinates
    of the feature vector to increase as well as decrease.
    """
    def __init__(self, input_dim, n_layer, n_head, d_model, d_head):
        super().__init__()
        self._input_dim = input_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._d_model = d_model
        self._d_head = d_head

        self._lin = tc.nn.Linear(
            self._input_dim, self._d_model)
        tc.nn.init.xavier_normal_(self._lin.weight)

        self._transformer_layers = tc.nn.ModuleList([
            TransformerLayer(
                d_model=self._d_model,
                n_head=self._n_head,
                d_head=self._d_head,
                d_ff=self._d_model,
                position_encoding_style='rel',
                attention_style='full',
                layer_ordering='nfa')
            for _ in range(0, self._n_layer)
        ])

        self._ln = LayerNorm(units=self._d_model)

    @property
    def output_dim(self):
        return self._d_model

    def initial_state(self, batch_size):
        return None

    def forward(self, inputs, prev_state=None):
        """
        Args:
            inputs: input vec tensor of shape [B, ..., I]
            prev_state: optional previous state.

         Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            output feature tensor of shape [B, ..., d_model]
            and new state
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        past_kvs = [None] * self._n_layer if prev_state is None else prev_state

        inputs = self._lin(inputs)
        inputs = tc.nn.ReLU()(inputs)

        new_kvs_by_layer = []
        for l in range(0, self._n_layer):
            inputs, new_kvs = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs_by_layer.append(new_kvs)

        features = self._ln(inputs)

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs_by_layer


class SparseTransformerXLI(tc.nn.Module):
    """
    Implements a Sparse Transformer (Child et al., 2019) variant,
    using the attention operations introduced by Dhariwal et al., 2020,
    and the relative position encoding from Dai et al., 2019,
    and the reordered layer ordering from Parisotto et al., 2019.
    """
    def __init__(self, input_dim, n_layer, n_head, d_model, d_head, n_context):
        super().__init__()
        self._input_dim = input_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._d_model = d_model
        self._d_head = d_head
        self._n_context = n_context
        self._attention_styles = ['row', 'column', 'previous_row']

        self._lin = tc.nn.Linear(
            self._input_dim, self._d_model, bias=False)
        tc.nn.init.xavier_normal_(self._lin.weight)

        self._transformer_layers = tc.nn.ModuleList([
            TransformerLayer(
                d_model=self._d_model,
                n_head=self._n_head,
                d_head=self._d_head,
                d_ff=self._d_model,
                position_encoding_style='rel',
                attention_style=self._attention_styles[l % 3],
                layer_ordering='nfa',
                row_len=self._row_len)
            for l in range(0, self._n_layer)
        ])

        self._ln = LayerNorm(units=self._d_model)

    @property
    def _row_len(self):
        small = math.floor(self._n_context ** 0.5)
        big = self._n_context // small
        while small * big != self._n_context:
            small -= 1
            big = self._n_context // small
        return small

    @property
    def output_dim(self):
        return self._d_model

    def initial_state(self, batch_size):
        return None

    def forward(self, inputs, prev_state=None):
        """
        Args:
            inputs: input vec tensor of shape [B, ..., I]
            prev_state: optional previous state.

         Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            output feature tensor of shape [B, ..., d_model]
            and new state
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        past_kvs = [None] * self._n_layer if prev_state is None else prev_state

        inputs = self._lin(inputs)
        inputs = tc.nn.ReLU()(inputs)

        new_kvs_by_layer = []
        for l in range(0, self._n_layer):
            inputs, new_kvs = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs_by_layer.append(new_kvs)

        features = self._ln(inputs)

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs_by_layer
