from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import OrderedDict

import math
import torch
from torch import Tensor, nn
from torch.nn import init
import torch.nn.functional as F

from monai.networks.blocks import SubpixelUpsample
from .vitdet import LayerNorm

class Lazy_Autoencoder(nn.Module):
    def __init__(self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_img_shape: Sequence[int],
    ):
        """
        Args:
            encoder: nn.Module with input shape (B, C, H, W)
            decoder: nn.Module with input shape (B, C, H', W') and output (B, C, H'', W'')
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_img_shape = latent_img_shape
        print('Reshape latent to ', self.latent_img_shape)
    
    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (tensor): Input image with shape (B, C, H, W)

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W)
        """
        latent, _ = self.encoder(x) #latent: (B,HW,C)
        latent_x = latent.transpose(1,2).reshape(self.latent_img_shape) #(B,HW,C)->(B,C,HW)->(B,C,H,W)
        out_img = self.decoder(latent_x)
        return out_img
    
    def forward_with_latent(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """forward with latent

        Args:
            x (tensor): Input image with shape (B, C, H, W)

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W)
            Output lantent: Output latent feature with shape (B,HW,C)
        """
        latent, _ = self.encoder(x)
        latent_x = latent.transpose(1,2).reshape(self.latent_img_shape) #(B,HW,C)->(B,C,HW)->(B,C,H,W)
        out_img = self.decoder(latent_x)
        return out_img, latent
    
class Conv_decoder(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        conv_bias: bool = True,
        use_layer_norm: bool = True,
        act_func: nn.Module = nn.GELU,
    ):
        """
        Args:
            in_channels: int, input hidden num
            out_channels: int, output channel num (1 or 3)
            scale_factor: int, scale factor from input to output, need to be 2^n
            conv_bias: bool = True, use bias in Conv layer or not
            use_layer_norm: bool = True, use layer norm or not
            act_func: nn.Module = nn.GELU, activation functions
        """
        super().__init__()
        if use_layer_norm:
            norm_func = LayerNorm
        else:
            norm_func = nn.Identity
        
        scale_factor_pow = int(math.log2(scale_factor)) #2^x of scale factor
        self.stages = nn.ModuleList()
        scale_w = scale_factor
        hidden_num = in_channels
        for stage in range(scale_factor_pow-1):
            hidden_num_out = int(max(hidden_num//2, 16))
            layers = [
                    nn.ConvTranspose2d(hidden_num, hidden_num_out, kernel_size=2, stride=2, bias=conv_bias),
                    norm_func(hidden_num_out,eps=1e-5,spatial_dims=2),
                    act_func(),
                    ]
            layers = nn.Sequential(*layers)
            self.stages.append(layers)
            hidden_num = hidden_num_out
            scale_w = int(max(hidden_num//2, 1)) 
        last_conv = nn.ConvTranspose2d(hidden_num, out_channels, kernel_size=2, stride=2, bias=conv_bias)
        self.stages.append(last_conv)
        
    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (tensor): Output from encoder with shape (B, C, H', W')

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W)
        """
        for stage in self.stages:
            x = stage(x)
        return x

###!!! not implement now
class Upsample_decoder(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        conv_bias: bool = False,
        use_layer_norm: bool = True,
        act_func: nn.Module = nn.LeakyReLU,
    ):
        """
        Args:
            in_channels: int, input hidden num
            out_channels: int, output channel num (1 or 3)
            scale_factor: int, scale factor from input to output, need to be 2^n
            conv_bias: bool = True, use bias in Conv layer or not
            use_layer_norm: bool = True, use layer norm or not
            act_func: nn.Module = nn.LeakyReLU, activation functions
        """
        super().__init__()
        if use_layer_norm:
            norm_func = LayerNorm
        else:
            norm_func = nn.Identity
        
        scale_factor_pow = int(math.log2(scale_factor)) #2^x of scale factor
        self.stages = nn.ModuleList()
        scale_w = scale_factor
        hidden_num = in_channels
        for stage in range(scale_factor_pow):
            hidden_num_out = int(max(hidden_num//2, 16))
            layers = [
                    #x: Tensor in shape (batch, channel, spatial_1[, spatial_2, …).
                    SubpixelUpsample(2,hidden_num, hidden_num_out, scale_factor=2, bias=conv_bias),
                    norm_func(hidden_num_out,eps=1e-5,spatial_dims=2),
                    act_func(),
                    ]
            layers = nn.Sequential(*layers)
            self.stages.append(layers)
            hidden_num = hidden_num_out
            scale_w = int(max(hidden_num//2, 1)) 
        last_conv = nn.Conv2d(hidden_num, out_channels, 3, 1, 1)
        self.stages.append(last_conv)
        
    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (tensor): Output from encoder with shape (B, C, H', W')

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W)
        """
        for stage in self.stages:
            x = stage(x)
        return x