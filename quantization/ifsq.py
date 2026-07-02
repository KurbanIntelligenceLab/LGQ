"""iFSQ: Improving FSQ for Image Generation with 1 Line of Code (arXiv:2601.17124).

The activation that bounds encoder outputs into the FSQ grid is changed from
``tanh(z)`` to ``2 * sigmoid(alpha * z) - 1`` with alpha=1.6, which maps
Gaussian-distributed activations to a uniform distribution on (-1, 1).
Everything else (levels, scale-and-shift, round-STE, projections) is inherited
from FSQ unchanged.
"""

from __future__ import annotations
from functools import partial

import torch
from torch import sigmoid, clamp

from external.vector_quantize_pytorch.finite_scalar_quantization import (
    FSQ,
    round_ste,
    identity,
)


class iFSQ(FSQ):
    """FSQ with sigmoid-alpha bounding (Gaussian -> uniform pre-quantization)."""

    def __init__(self, *args, alpha: float = 1.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def bound(self, z, eps: float = 1e-3, hard_clamp: bool = False):
        alpha = self.alpha
        if hard_clamp:
            act = partial(clamp, min=-1., max=1.)
            inv = identity
        else:
            def act(x):
                return 2.0 * sigmoid(alpha * x) - 1.0

            def inv(y):
                # inverse of act:  y = 2*sigmoid(a*x) - 1  =>  x = logit((y+1)/2) / a
                return torch.log((1. + y) / (1. - y)) / alpha

        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = inv(offset / half_l)
        bounded_z = act(z + shift) * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width
