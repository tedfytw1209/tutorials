'''
Detection Metrics Function implement through MONAI. COCOMetrics
'''
import torch
from torch import Tensor
import numpy as np
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Dict
from monai.metrics import PSNRMetric as MonaiPSNRMetric
from monai.losses import SSIMLoss as MonaiSSIMLoss

class PSNRMetric:
    def __init__(self, max_val=1.0):
        self.metric = MonaiPSNRMetric(max_val=max_val)

    def __call__(self, y_pred, y):
        self.metric(y_pred=y_pred, y=y)
        out = self.metric.aggregate()
        self.reset()
        return out

    def aggregate(self):
        return self.metric.aggregate()

    def reset(self):
        self.metric.reset()

class SSIMLoss:
    def __init__(self, spatial_dims: int = 2,data_range: float = 1.0,win_size: int=11):
        self.loss = MonaiSSIMLoss(spatial_dims=spatial_dims,data_range=data_range,win_size=win_size)

    def __call__(self, y_pred, y):
        return self.loss(y_pred, y)

def Peak_Signal_Noise_Ratio(
    outputs: Tensor, 
    targets: Tensor,
    data_range: float,
    ):
    '''
    Args:
        outputs: Tensor, output image with shape (B,C,H,W)
        targets: Tensor, target origin image with shape (B,C,H,W)
        data_range: float, data range of value 1.0 or 255
    Return:
        PSNR value: Tensor
    '''
    psnr = MonaiPSNRMetric(max_val=data_range)
    return psnr(outputs, targets)

def Structural_Similarity(
    outputs: Tensor, 
    targets: Tensor,
    spatial_dims: int = 2,
    data_range: float = 1.0,
    win_size: int=11,
    ):
    '''
    Args:
        outputs: Tensor, output image with shape (B,C,H,W)
        targets: Tensor, target origin image with shape (B,C,H,W)
        spatial_dims: int, 2 or 3
        data_range: float = 1.0, data range of value 1.0 or 255
        win_size: int, window size in ssim
    Return:
        SSIM value: Tensor
    '''
    ssim = MonaiSSIMLoss(spatial_dims,data_range)
    return ssim(outputs,targets)