from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import OrderedDict

import math
import torch
from torch import Tensor, nn
from torch.nn import init
import torch.nn.functional as F

from timm.models.layers import DropPath

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.mlp import MLPBlock
from monai.utils import optional_import
from monai.networks.nets import ViT


class Lazy_Autoencoder(nn.Module):
    def __init__(self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """forward

        Args:
            x (tensor): Input image with shape (B, C, H, W, D) or (B, C, H, W)

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W, D) or (B, C, H, W)
            Output lantent: Output latent feature with shape (B,HW,C)
        """
        latent, _ = self.encoder(x)
        out_img = self.decoder(latent)
        return out_img, latent
    
class Conv_decoder(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        conv_block: str ='default',
        apply_pad_pool: bool = True,
        conv_bias: bool = True,
        use_layer_norm: bool = True,
        act_func: nn.Module = nn.GELU,
    ):
        super().__init__()
        scale_factor_pow = int(math.log2(scale_factor)) #2^x of scale factor
        self.stages = nn.ModuleList()
        scale_w = scale_factor
        hidden_num = in_channels
        for stage in range(scale_factor_pow-1):
            hidden_num = max(hidden_num/2, 16) 
            pass
        last_conv = nn.ConvTranspose2d(hidden_num, out_channels=out_channels, kernel_size=2, stride=2)
        self.stages.append(last_conv)
            
        
    def forward(self, x: Tensor) -> Tensor:
        pass