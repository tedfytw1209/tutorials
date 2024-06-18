from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import OrderedDict

import math
import torch
from torch import Tensor, nn
from torch.nn import init
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape: int, eps: float=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.reset_parameters() #self initialize

    def forward(self, x):
        #print('Layer Norm input shape: ', x.shape)
        u = x.detach().mean(1, keepdim=True)
        s = (x.detach() - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (torch.sqrt(s + self.eps) + self.eps)
        #print('Layer Norm before wx+b shape: ', x.shape)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        #print('Layer Norm output shape: ', x.shape, 'wieght: ', self.weight.shape, 'bias: ', self.bias.shape)
        return x

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)

class Lazy_Autoencoder(nn.Module):
    def __init__(self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_img_shape: Sequence[int],
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_img_shape = latent_img_shape
        print('Reshape latent to ', self.latent_img_shape)
    
    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (tensor): Input image with shape (B, C, H, W, D) or (B, C, H, W)

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W, D) or (B, C, H, W)
        """
        latent, _ = self.encoder(x) #latent: (B,HW,C)
        latent_x = latent.transpose(1,2).reshape(self.latent_img_shape) #(B,HW,C)->(B,C,HW)->(B,C,H,W)
        out_img = self.decoder(latent_x)
        return out_img
    
    def forward_with_latent(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """forward

        Args:
            x (tensor): Input image with shape (B, C, H, W, D) or (B, C, H, W)

        Returns:
            Output x: Output super resolution image with shape (B, C, H, W, D) or (B, C, H, W)
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
                    norm_func(hidden_num_out,eps=1e-5),
                    act_func(),
                    ]
            layers = nn.Sequential(*layers)
            self.stages.append(layers)
            hidden_num = hidden_num_out
            scale_w = int(max(hidden_num//2, 1)) 
        last_conv = nn.ConvTranspose2d(hidden_num, out_channels, kernel_size=2, stride=2, bias=conv_bias)
        self.stages.append(last_conv)
        
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
        return x

###!!! not implement now
class UNETR_decoder(nn.Module):
    pass