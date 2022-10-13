# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch

_gpu_is_old: Optional[bool] = None


def gpu_capabilities_older_than_70() -> bool:
    """Return True if the GPU's compute capability is older than SM70."""
    global _gpu_is_old
    if _gpu_is_old is None:
        for i in range(torch.cuda.device_count()):
            major, _ = torch.cuda.get_device_capability(f"cuda:{i}")
            if major < 7:
                _gpu_is_old = True
        if _gpu_is_old is None:
            _gpu_is_old = False
    return _gpu_is_old


SUPPORTED_CUDA_DEVICES = ["V100", "A100", "T4"]


def get_current_cuda_device():
    current_device = str(torch.cuda.get_device_properties(torch.cuda.current_device()))
    for device_str in SUPPORTED_CUDA_DEVICES:
        if current_device.find(device_str) > 0:
            return device_str

    logging.warning("Unsupported device, Triton code generation may fail")
    return "P100"  # default to an old GPU


def assert_almost_equal(x, y, decimal=2, err_msg=""):
    import numpy.testing as npt

    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)


def assert_allclose(x, y, rtol=1e-07, atol=0, err_msg=""):
    import numpy.testing as npt

    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    npt.assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)
