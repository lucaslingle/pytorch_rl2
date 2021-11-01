"""
Implements SNAIL architecture (Mishra et al., 2017) for RL^2.
"""

import torch as tc


class TCBlock(tc.nn.Module):
    def __init__(self, input_dim, feature_dim, dilation_rate, context_size):
        super().__init__()
        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._dilation_rate = dilation_rate
        self._context_size = context_size

        # TODO(lucaslingle):
        # implement CausalConv, DenseBlock,
        # and put a ModuleList of DenseBlocks here
        self._dense_blocks = tc.nn.ModuleList()

    @property
    def num_layers(self):
        raise NotImplementedError

    def forward(self, input_vec, prev_state):
        present_activations = input_vec.unsqueeze(1)  # [B, 1, T, F]
        past_activations = prev_state  # [B, L, T, F]
        num_layers = self.num_layers()
        for l in num_layers:
            combined_layer_input = tc.cat(
                (past_activations[0:(l-1)], present_activations[0:(l-1)]),
                dim=2)
            output = self._dense_blocks[l](combined_layer_input)
            present_activations = tc.cat(
                (present_activations, output),
                dim=1)
        return present_activations

