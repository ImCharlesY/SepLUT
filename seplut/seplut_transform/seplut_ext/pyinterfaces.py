from typing import Tuple

import torch
from torch.cuda.amp import custom_fwd, custom_bwd

from ._ext import cforward


class SepLUTTransformFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                img: torch.Tensor,
                lut3d: torch.Tensor,
                lut1d: torch.tensor) -> torch.Tensor:

        img = img.contiguous()
        lut3d = lut3d.contiguous()
        lut1d = lut1d.contiguous()

        assert img.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut3d.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        assert lut1d.ndimension() == 3, \
            "only support 1D lookup table with batch dimension (3D tensor)"

        output = img.new_zeros((img.size(0), lut3d.size(1), img.size(2), img.size(3)))
        cforward(img, lut3d, lut1d, output)

        ctx.save_for_backward(img, lut3d, lut1d)

        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.contiguous()

        img, lut3d, lut1d = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut3d = torch.zeros_like(lut3d)
        grad_lut1d = torch.zeros_like(lut1d)

        return grad_img, grad_lut3d, grad_lut1d
    
    
def seplut_transform(
    img: torch.Tensor,
    lut3d: torch.Tensor,
    lut1d: torch.Tensor) -> torch.Tensor:
    r"""Adaptive Interval 3D Lookup Table Transform (SepLUT-Transform).
    
    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut3d (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
        lut1d (torch.Tensor): output values of the 1D LUT, shape (b, 3, m).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return SepLUTTransformFunction.apply(img, lut3d, lut1d)
