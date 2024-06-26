'''
Detection Metrics Function implement through MONAI. COCOMetrics
'''
import torch
from torch import Tensor
import numpy as np
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Dict

from torcheval.metrics.image import PeakSignalNoiseRatio
from torcheval.metrics.image.ssim import StructuralSimilarity
from skimage.metrics import structural_similarity
from typing import Iterable, Optional, TypeVar
TStructuralSimilarity = TypeVar("TStructuralSimilarity")

class StructuralSimilarity_gray(StructuralSimilarity):
    '''
    Code from torcheval.metrics.image.ssim.StructuralSimilarity and modify for gray image
    '''
    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TStructuralSimilarity,
        images_1: torch.Tensor,
        images_2: torch.Tensor,
    ) -> TStructuralSimilarity:
        """
        Update the metric state with new input.
        Ensure that the two sets of images have the same value range (ex. [-1, 1], [0, 1]).

        Args:
            images_1 (Tensor): A batch of the first set of images of shape [N, C, H, W].
            images_2 (Tensor): A batch of the second set of images of shape [N, C, H, W].

        """
        if images_1.shape != images_2.shape:
            raise RuntimeError("The two sets of images must have the same shape.")
        # convert to fp32, mostly for bf16 types
        images_1 = images_1.to(dtype=torch.float32)
        images_2 = images_2.to(dtype=torch.float32)

        batch_size = images_1.shape[0]

        for idx in range(batch_size):
            mssim = structural_similarity(
                images_1[idx].squeeze().detach().cpu().numpy(),
                images_2[idx].squeeze().detach().cpu().numpy(),
                multichannel=False,
                data_range=1.0,
            )
            self.mssim_sum += mssim

        self.num_images += batch_size

        return self

class PSNR():
    def __init__(self, data_range: float, device: str):
        self.psnr = PeakSignalNoiseRatio(data_range=data_range, device=device)

    def __call__(self, outputs: Tensor, targets: Tensor):
        self.psnr.update(outputs, targets)
        return self.psnr.compute()
    
class SSIM():
    def __init__(self, device: str):
        self.ssim = StructuralSimilarity_gray(device=device)

    def __call__(self, outputs: Tensor, targets: Tensor):
        self.ssim.update(outputs, targets)
        return self.ssim.compute()

def Peak_Signal_Noise_Ratio(
    outputs: Tensor, 
    targets: Tensor,
    data_range: float,
    device: str,
    ):
    '''
    Args:
        outputs: Tensor, output image with shape (B,C,H,W)
        targets: Tensor, target origin image with shape (B,C,H,W)
        data_range: float, data range of value 1.0 or 255
        device: str, used device
    Return:
        PSNR value: Tensor
    '''
    psnr = PSNR(data_range=data_range, device=device)
    return psnr(outputs, targets)

def Structural_Similarity(
    outputs: Tensor, 
    targets: Tensor,
    device: str,
    ):
    '''
    Args:
        outputs: Tensor, output image with shape (B,C,H,W)
        targets: Tensor, target origin image with shape (B,C,H,W)
        device: str, used device
    Return:
        SSIM value: Tensor
    '''
    ssim = SSIM(device=device)
    return ssim(outputs,targets)