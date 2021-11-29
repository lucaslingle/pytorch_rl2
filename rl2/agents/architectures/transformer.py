"""
Implements Transformer architectures for RL^2.
"""

import torch as tc

from rl2.agents.architectures.common.normalization import LayerNorm
from rl2.agents.architectures.common.attention import (
    MultiheadSelfAttention,
    sinusoidal_embeddings
)


class FF(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            activation=tc.nn.ReLU
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

        self._act = activation()

        self._lin2 = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._output_dim,
            bias=True)
        tc.nn.init.xavier_normal_(self._lin2.weight)
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
        x = self._act(x)
        x = self._lin2(x)
        return x


class TransformerLayer(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            num_heads,
            position_encoding_style,
            attention_style,
            connection_style,
            layer_ordering,
            row_len=None,
            activation=tc.nn.ReLU
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
            activation: activation function to use in ff and anywhere else.
        """
        assert position_encoding_style in ['abs', 'rel']
        assert attention_style in ['full', 'row', 'previous_row', 'column']
        assert connection_style in ['plain', 'residual', 'dense']
        assert attention_style == 'full' or row_len is not None
        assert len(layer_ordering) == len(set(layer_ordering))
        assert set(layer_ordering) <= {'a', 'f', 'n'}
        assert 'f' in set(layer_ordering)

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
        self._activation = activation

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
            output_dim=self._feature_dim,
            activation=self._activation)

        if 'n' in self._layer_ordering:
            if self._layer_ordering.index('n') < self._layer_ordering.index('f'):
                self._attn_layer_norm = LayerNorm(units=self._attn_input_dim)
                self._ff_layer_norm = LayerNorm(units=self._ff_input_dim)
            else:
                self._attn_layer_norm = LayerNorm(units=self._feature_dim)
                self._ff_layer_norm = LayerNorm(units=self._feature_dim)

        if 'a' in self._layer_ordering:
            self._attn_act = self._activation()
            self._ff_act = self._activation()

    @property
    def _attn_input_dim(self):
        return self._input_dim

    @property
    def _ff_input_dim(self):
        if self._connection_style != 'dense':
            return self._feature_dim
        return self._attn_input_dim + self._feature_dim

    @property
    def output_dim(self):
        if self._connection_style != 'dense':
            return self._feature_dim
        return self._ff_input_dim + self._feature_dim

    def forward(self, inputs, past_kvs=None):
        """
        Args:
            inputs: input vec tensor of shape [B, T2, I]
            past_kvs: optional past kvs

        Returns:
            output tensor of shape [B, T2, I], and new kvs
        """
        x = inputs

        i = inputs
        for letter in self._layer_ordering:
            if letter == 'a':
                x = self._attn_act(x)
            elif letter == 'n':
                x = self._attn_layer_norm(x)
            elif letter == 'f':
                x, new_kvs = self._attn(x, past_kvs=past_kvs)
                x = self._proj(x)
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


class Transformer(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            feature_dim,
            n_layer,
            n_head,
            n_context,
            position_encoding_style='abs',
            attention_style='sparse',
            connection_style='dense',
            layer_ordering='fn',
            input_logic='',
            output_logic='',
            activation=tc.nn.ReLU
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
            connection_style: one of 'plain', 'residual', 'dense'.
            layer_ordering: ordering of activation, function, and normalization.
                string should be letters chosen from 'a', 'f', 'n' in any order.
                letter 'f' cannot be omitted.
            input_logic: ordering of activation and normalization after
                input linear projection and prior to first transformer layer.
                string should be letters chosen from 'a', 'n' in any order.
                string defaults to empty.
            output_logic: ordering of activation and normalization before features
                are returned and after last transformer layer.
                string should be letters chosen from 'a', 'n' in any order.
                string defaults to empty.
            activation: activation function to use in ff and anywhere else.
        """
        assert position_encoding_style in ['abs', 'rel']
        assert attention_style in ['full', 'sparse']
        assert connection_style in ['residual', 'dense']
        assert len(layer_ordering) == len(set(layer_ordering))
        assert set(layer_ordering) <= {'a', 'f', 'n'}
        assert 'f' in set(layer_ordering)
        assert len(input_logic) == len(set(input_logic))
        assert set(input_logic) <= {'a', 'n'}
        assert len(output_logic) == len(set(output_logic))
        assert set(output_logic) <= {'a', 'n'}

        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._n_context = n_context
        self._position_encoding_style = position_encoding_style
        self._connection_style = connection_style
        self._layer_ordering = list(layer_ordering)
        self._input_logic = list(input_logic)
        self._output_logic = list(output_logic)
        self._activation = activation

        # input
        self._input_proj = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=self._feature_dim)
        tc.nn.init.xavier_normal_(self._input_proj.weight)
        if 'n' in self._input_logic:
            self._input_layer_norm = LayerNorm(units=self._feature_dim)
        if 'a' in self._input_logic:
            self._input_act = self._activation()

        if self._position_encoding_style == 'abs':
            self._position_embeddings = sinusoidal_embeddings(
                self._n_context, self._feature_dim, reverse=False)

        # middle
        self._transformer_layers = tc.nn.ModuleList([
            TransformerLayer(
                input_dim=self._get_input_dim(l),
                feature_dim=self._feature_dim,
                num_heads=self._n_head,
                position_encoding_style=self._position_encoding_style,
                attention_style=self._get_attention_style(attention_style, l),
                connection_style=self._connection_style,
                layer_ordering=''.join(self._layer_ordering),
                row_len=self._get_row_len(attention_style),
                activation=self._activation)
            for l in range(self._n_layer)
        ])

        # output
        if 'n' in self._output_logic:
            self._output_layer_norm = LayerNorm(units=self.output_dim)
        if 'a' in self._output_logic:
            self._output_act = self._activation()

    def _get_input_dim(self, l):
        if self._connection_style != 'dense':
            return self._feature_dim
        else:
            if self._position_encoding_style == 'abs':
                return (2*l+2) * self._feature_dim
            return (2*l+1) * self._feature_dim

    def _get_attention_style(self, attention_style, l):
        if attention_style == 'full':
            return 'full'
        sparse_attention_styles = ['row', 'column', 'previous_row']
        return sparse_attention_styles[l % 3]

    def _get_row_len(self, attention_style):
        if attention_style == 'full':
            return None
        small = int(self._n_context ** 0.5)
        while self._n_context % small != 0:
            small -= 1
        return small

    def _get_past_len(self, prev_state):
        assert prev_state is None or isinstance(prev_state, list)
        if prev_state is None:
            return 0
        k, _ = prev_state[0]  # layer 0, get keys
        if isinstance(k, list):
            return len(k)
        if isinstance(k, tc.Tensor):
            return k.shape[1]
        raise NotImplementedError

    def _add_position_embeddings(self, inputs, prev_state):
        t1 = self._get_past_len(prev_state)
        t2 = inputs.shape[1]
        assert t1 + t2 <= self._n_context
        pos_embs = self._position_embeddings[t1:t1+t2, :]
        pos_embs = pos_embs.unsqueeze(0)
        if self._connection_style != 'dense':
            inputs = inputs + pos_embs
        else:
            pos_embs = tc.tile(pos_embs, [inputs.shape[0], 1, 1])
            inputs = tc.cat((inputs, pos_embs), dim=-1)
        return inputs

    def _run_input_logic(self, inputs):
        for letter in self._input_logic:
            if letter == 'n':
                inputs = self._input_layer_norm(inputs)
            elif letter == 'a':
                inputs = self._input_act(inputs)
        return inputs

    def _run_output_logic(self, inputs):
        for letter in self._output_logic:
            if letter == 'n':
                inputs = self._output_layer_norm(inputs)
            elif letter == 'a':
                inputs = self._output_act(inputs)
        return inputs

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

        # input
        inputs = self._input_proj(inputs)
        inputs = self._run_input_logic(inputs)
        if self._position_encoding_style == 'abs':
            inputs = self._add_position_embeddings(inputs, prev_state)

        # middle
        past_kvs = [None] * self._n_layer if prev_state is None else prev_state
        new_kvs = []
        for l in range(0, self._n_layer):
            inputs, new_kvs_l = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs.append(new_kvs_l)

        # output
        inputs = self._run_output_logic(inputs)

        features = inputs
        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs
