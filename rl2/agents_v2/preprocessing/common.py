"""
Implements common agent components used in Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

import torch as tc


def one_hot(ys: tc.LongTensor, depth: int) -> tc.FloatTensor:
    """
    Applies one-hot encoding to a batch of vectors.

    Args:
        ys: tc.LongTensor of shape [B].
        depth: int specifying the number of possible y values.

    Returns:
        the one-hot encodings of tensor ys.
    """

    vecs_shape = list(ys.shape) + [depth]
    vecs = tc.zeros(dtype=tc.float32, size=vecs_shape)
    vecs.scatter_(dim=-1, index=ys.unsqueeze(-1),
                  src=tc.ones(dtype=tc.float32, size=vecs_shape))
    return vecs.float()
