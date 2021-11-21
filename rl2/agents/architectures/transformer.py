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

        features = inputs

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

        features = inputs

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs_by_layer


class DCTransformerLayer(tc.nn.Module):
    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            num_heads: int,
            position_encoding_style: str,
            attention_style: str,
            row_len: Optional[int] = None
    ):
        """
        Densely-connected transformer layer.

        Args:
            input_dim: input dimensionality.
            feature_dim: feature dimensionality for each sublayer.
            num_heads: number of attention heads.
            position_encoding_style: one of 'abs', 'rel'.
            attention_style: one of 'full', 'row', 'column', or 'previous_row'.
            row_len: required if attention_style is not 'full'
        """
        assert attention_style == 'full' or row_len is not None
        assert feature_dim % num_heads == 0

        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._num_heads = num_heads
        self._num_features_per_head = self._feature_dim // self._num_heads
        self._position_encoding_style = position_encoding_style
        self._attention_style = attention_style
        self._row_len = row_len

        self._attn_act = tc.nn.ReLU()
        self._ff_act = tc.nn.ReLU()

        self._attn_layer_norm = LayerNorm(units=self._feature_dim)
        self._ff_layer_norm = LayerNorm(units=self._feature_dim)

        self._attn = MultiheadSelfAttention(
            input_dim=self._input_dim,
            num_heads=self._num_heads,
            num_head_features=self._num_features_per_head,
            position_encoding_style=self._position_encoding_style,
            attention_style=self._attention_style,
            row_len=self._row_len)
        self._proj = tc.nn.Linear(
            in_features=(self._num_heads * self._num_features_per_head),
            out_features=self._feature_dim,
            bias=False)
        tc.nn.init.xavier_normal_(self._proj.weight)

        self._ff = FF(
            input_dim=(self._input_dim + self._feature_dim),
            hidden_dim=self._feature_dim,
            output_dim=self._feature_dim)

    @property
    def output_dim(self):
        return self._input_dim + 2 * self._feature_dim

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
        x, new_kvs = self._attn(inputs=x, past_kvs=past_kvs)
        x = self._proj(x)
        x = self._attn_layer_norm(x)
        x = self._attn_act(x)
        x = tc.cat((i, x), dim=-1)

        i = x
        x = self._ff(x)
        x = self._ff_layer_norm(x)
        x = self._ff_act(x)
        x = tc.cat((i, x), dim=-1)

        return x, new_kvs


class DCSparseTransformerXL(tc.nn.Module):
    """
    Implements a densely-connected version of a Sparse Transformer variant,
    using the attention operations introduced by Dhariwal et al., 2020,
    and the relative position encoding from Dai et al., 2019.
    """
    def __init__(self, input_dim, feature_dim, n_layer, n_head, n_context):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._n_context = n_context
        self._attention_styles = ['row', 'column', 'previous_row']

        self._input_proj = tc.nn.Linear(
             in_features=self._input_dim,
             out_features=self._feature_dim)
        tc.nn.init.xavier_normal_(self._input_proj.weight)
        self._input_layer_norm = LayerNorm(units=self._feature_dim)
        self._input_act = tc.nn.ReLU()

        self._transformer_layers = tc.nn.ModuleList([
            DCTransformerLayer(
                input_dim=((2*l+1) * self._feature_dim),
                feature_dim=self._feature_dim,
                num_heads=self._n_head,
                position_encoding_style='rel',
                attention_style=self._attention_styles[l % 3],
                row_len=self._row_len)
            for l in range(0, self._n_layer)
        ])

    @property
    def _row_len(self):
        small = math.floor(self._n_context ** 0.5)
        while self._n_context % small != 0:
            small -= 1
        return small

    @property
    def output_dim(self):
        return self._transformer_layers[-1].output_dim

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
            output feature tensor and new state.
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        past_kvs = [None] * self._n_layer if prev_state is None else prev_state

        inputs = self._input_proj(inputs)
        inputs = self._input_layer_norm(inputs)
        inputs = self._input_act(inputs)

        new_kvs_by_layer = []
        for l in range(0, self._n_layer):
            inputs, new_kvs = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs_by_layer.append(new_kvs)

        features = inputs

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs_by_layer


class NeoTransformerLayer(tc.nn.Module):
    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            num_heads: int,
            position_encoding_style: str,
            attention_style: str,
            connection_style: str,
            layer_ordering: str,
            row_len: Optional[int] = None
    ):
        """
        Args:
            input_dim: input dimensionality.
            feature_dim: feature dimensionality for each sublayer.
            num_heads: number of attention heads.
            position_encoding_style: one of 'abs', 'rel'.
            attention_style: one of 'full', 'row', 'column', or 'previous_row'.
            connection_style: one of 'plain', 'residual', 'dense'.
            layer_ordering: ordering of activation, function, and normalization.
                should be letters chosen from 'a', 'f', 'n' in any order.
                letter 'f' cannot be omitted.
            row_len: required if attention_style is not 'full'
        """
        assert position_encoding_style in ['abs', 'rel']
        assert attention_style in ['full', 'row', 'previous_row', 'column']
        assert connection_style in ['plain', 'residual', 'dense']
        assert attention_style == 'full' or row_len is not None
        assert len(list(layer_ordering)) <= 3
        assert set(list(layer_ordering)) <= {'a', 'f', 'n'}
        assert 'f' in list(layer_ordering)

        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._num_heads = num_heads
        self._num_features_per_head = self._feature_dim // self._num_heads
        self._position_encoding_style = position_encoding_style
        self._attention_style = attention_style
        self._connection_style = connection_style
        self._layer_ordering = list(layer_ordering)
        self._row_len = row_len

        self._attn = MultiheadSelfAttention(
            input_dim=self._attn_input_dim,
            num_heads=self._num_heads,
            num_head_features=self._num_features_per_head,
            position_encoding_style=self._position_encoding_style,
            attention_style=self._attention_style,
            row_len=self._row_len)
        self._proj = tc.nn.Linear(
            in_features=(self._num_heads * self._num_features_per_head),
            out_features=self._feature_dim,
            bias=False)
        tc.nn.init.xavier_normal_(self._proj.weight)

        self._ff = FF(
            input_dim=self._ff_input_dim,
            hidden_dim=self._feature_dim,
            output_dim=self._feature_dim)

        if 'n' in self._layer_ordering:
            if self._layer_ordering.index('n') < self._layer_ordering.index('f'):
                self._attn_layer_norm = LayerNorm(units=self._attn_input_dim)
                self._ff_layer_norm = LayerNorm(units=self._ff_input_dim)
            else:
                self._attn_layer_norm = LayerNorm(units=self._feature_dim)
                self._ff_layer_norm = LayerNorm(units=self._feature_dim)

        if 'a' in self._layer_ordering:
            self._attn_act = tc.nn.ReLU()
            self._ff_act = tc.nn.ReLU()

    @property
    def _attn_input_dim(self):
        return self._input_dim

    @property
    def _ff_input_dim(self):
        if self._connection_style != 'dense':
            return self._feature_dim
        return self._attn_sublayer_in_dim + self._feature_dim

    def forward(self, inputs, past_kvs=None):
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
                x = attn_output
                if self._connection_style == 'residual':
                    x += i
            else:
                raise NotImplementedError

        if self._connection_style == 'dense':
            x = tc.cat((i, x), dim=-1)

        i = x
        for letter in self._layer_ordering:
            if letter == 'a':
                x = self._ff_act(x)
            elif letter == 'n':
                x = self._ff_layer_norm(x)
            elif letter == 'f':
                x = self._ff(x)
                if self._connection_style == 'residual':
                    x += i
            else:
                raise NotImplementedError

        if self._connection_style == 'dense':
            x = tc.cat((i, x), dim=-1)

        return x, new_kvs


class NeoTransformer(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            n_layer,
            n_head,
            n_context,
            position_encoding_style,
            attention_style,
            connection_style,
            layer_ordering,
    ):
        """
        Args:
            input_dim: input dimensionality.
            feature_dim: feature dimensionality for each sublayer.
            n_layer: number of transformer layers.
            n_head: number of attention heads.
            n_context: meta-episode length.
            position_encoding_style: one of 'abs', 'rel'.
            attention_style: one of 'full', 'sparse'.
            connection_style: one of 'residual', 'dense'.
            layer_ordering: ordering of activation, function, and normalization.
                should be letters chosen from 'a', 'f', 'n' in any order.
                letter 'f' cannot be omitted.
        """
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._n_context = n_context
        self._position_encoding_style = position_encoding_style
        self._attention_styles = self._get_attention_styles(attention_style)
        self._connection_style = connection_style
        self._layer_ordering = list(layer_ordering)

        # input
        self._input_proj = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=self._feature_dim,
            bias=False)
        tc.nn.init.xavier_normal_(self._input_proj.weight)
        if 'n' in self._layer_ordering:
            if self._layer_ordering.index('f') < self._layer_ordering.index('n'):
               self._input_layer_norm = LayerNorm(units=self._feature_dim)
        if 'a' in self._layer_ordering:
            self._input_act = tc.nn.ReLU()

        # middle
        self._transformer_layers = tc.nn.ModuleList([
            NeoTransformerLayer(
                input_dim=self._get_input_dim(l),
                feature_dim=self._feature_dim,
                num_heads=self._n_head,
                position_encoding_style=self._position_encoding_style,
                attention_style=self._get_attention_style(attention_style, l),
                connection_style=self._connection_style,
                layer_ordering=''.join(self._layer_ordering),
                row_len=self._get_row_len(attention_style))
            for l in range(self._n_layer)
        ])

        # output
        if 'n' in self._layer_ordering:
            if self._layer_ordering.index('n') < self._layer_ordering.index('f'):
               self._output_layer_norm = LayerNorm(units=self.output_dim)

        if 'a' in self._layer_ordering:
            if self._layer_ordering.index('a') < self._layer_ordering.index('f'):
               self._output_act = tc.nn.ReLU()

    def _get_input_dim(self, l):
        if self._connection_style != 'dense':
            return self._feature_dim
        return (2*l+1) * self._feature_dim

    def _get_attention_style(self, attention_style, l):
        if attention_style == 'full':
            return 'full'

        sparse_attention_styles = ['row', 'column', 'previous_row']
        return sparse_attention_styles[l % 3]

    def _row_len(self, attention_style):
        if attention_style == 'full':
            return None
        small = math.floor(self._n_context ** 0.5)
        while self._n_context % small != 0:
            small -= 1
        return small

    @property
    def output_dim(self):
        return self._transformer_layers[-1].output_dim

    def initial_state(self, batch_size):
        return None

    def forward(self, inputs, prev_state=None):
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        past_kvs = [None] * self._n_layer if prev_state is None else prev_state

        # input
        inputs = self._input_proj(inputs)
        for letter in self._layer_ordering:
            if letter == 'n':
                if self._layer_ordering.index('f') < self._layer_ordering.index('n'):
                    inputs = self._input_layer_norm(inputs)
                else:
                    continue
            elif letter == 'a':
                inputs = self._input_act(inputs)
            else:
                continue

        # middle
        new_kvs_by_layer = []
        for l in range(0, self._n_layer):
            inputs, new_kvs = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs_by_layer.append(new_kvs)

        # output
        for letter in self._layer_ordering:
            if letter == 'n':
                if self._layer_ordering.index('n') < self._layer_ordering.index('f'):
                    inputs = self._output_layer_norm(inputs)
                else:
                    continue
            elif letter == 'a':
                if self._layer_ordering.index('a') < self._layer_ordering.index('f'):
                    inputs = self._output_act(inputs)
                else:
                    continue
            else:
                continue

        features = inputs

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs_by_layer
