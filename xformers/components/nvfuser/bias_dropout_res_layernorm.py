# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from functorch.compile import memory_efficient_fusion

from xformers.components import LayerNormStyle


def _fn(
    x: torch.Tensor,
    bias: torch.Tensor,
    prob: float,
    layer_norm_style: LayerNormStyle,
    norm: nn.Module,
    orig: torch.Tensor,
) -> torch.Tensor:
    if bias is not None:
        a = torch.add(x, bias)
    b = torch.nn.functional.dropout(a, prob) if prob > 0.0 else a
    if layer_norm_style == LayerNormStyle.Pre:
        c = norm(b)
        return torch.add(c, orig)
    elif layer_norm_style == LayerNormStyle.Post:
        c = torch.add(b, orig)
        return norm(c)
    else:
        raise ValueError


class NVFusedBiasDropoutResLayerNorm(torch.nn.Module):

    """
    A layer which fuses the computation of LayerNorm(Dropout(x + Bias) + Residual)
    with AOTAutograd and nvFuser
    """

    def __init__(
        self,
        p: float,
        d_model: int,
        bias_shape: Optional[int] = None,
        layer_norm_style: LayerNormStyle = LayerNormStyle.Post,
    ) -> None:
        super().__init__()

        self.p = float(p)
        self.layer_norm_style = layer_norm_style

        assert (
            self.p < 1.0
        ), f"We don't want to drop all the values, most probably p={self.p} is not properly set"

        self.bias = (
            nn.Parameter(torch.zeros(bias_shape, device=torch.device("cuda")))
            if bias_shape is not None
            else None
        )

        self.norm = nn.LayerNorm(d_model)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.bias is not None:
                self.bias.fill_(0.0)

    def forward(self, x: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
        # Train/inference
        p = self.p if self.training else 0.0

        # Catch a non-cuda setup, fallback to pytorch
        if not x.is_cuda or p == 0.0:
            return _fn(x, self.bias, p, self.layer_norm_style, self.norm, orig)

        # AOTAutograd, NVFuser backed path
        aot_fn = memory_efficient_fusion(fn=_fn, static_argnums=(2, 3, 4))
        return aot_fn(x, self.bias, p, self.layer_norm_style, self.norm, orig)