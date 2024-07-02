# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Sequence
from collections import OrderedDict
from typing import Union

import math
import torch
from torch import Tensor, nn
from torch.nn import init
import torch.nn.functional as F

from timm.models.layers import DropPath

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.mlp import MLPBlock
from monai.utils import optional_import
#from monai.networks.nets import ViT
#from monai.networks.layers.factories import Conv, Pool

from dataclasses import dataclass
from typing import Optional

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")
_validate_trainable_layers, _ = optional_import(
    "torchvision.models.detection.backbone_utils", name="_validate_trainable_layers"
)
torchvision_models, _ = optional_import("torchvision.models")

"""
This module implements SimpleFeaturePyramid in :paper:`vitdet`.
Code come from https://github.com/facebookresearch/detectron2.git implement
It creates pyramid features built on top of the input feature map.
"""

@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None

def window_partition(x: Tensor, window_size: int):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = int((window_size - H % window_size) % window_size)
    pad_w = int((window_size - W % window_size) % window_size)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [0, 0, 0, pad_w, 0, pad_h])
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows: Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(attn: Tensor, q: Tensor, rel_pos_h: Tensor, rel_pos_w: Tensor, q_size: tuple[int, int], k_size: tuple[int, int]):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

#From https://github.com/OpenGVLab/InternVL/blob/main/classification/models/intern_vit_6b.py#L127
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

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
        u = x.detach().mean(1, keepdim=True)
        s = (x.detach() - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / (torch.sqrt(s + self.eps) + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)

class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        dim_head: int | None = None,
        use_rel_pos: bool =False,
        rel_pos_zero_init: bool =True,
        input_size: Sequence[int] | int | None = None,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_normalization: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            dim_head (int, optional): dimension of each head. Defaults to hidden_size // num_heads.
            rel_pos (bool): If True, add relative positional embeddings to the attention map. !!!not impl
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads if dim_head is None else dim_head
        self.inner_dim = self.dim_head * num_heads

        self.proj = nn.Linear(self.inner_dim, hidden_size)
        self.qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.scale = self.dim_head**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()
        self.q_norm = norm_layer(hidden_size) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(hidden_size) if qk_normalization else nn.Identity()
        self.qk_normalization = qk_normalization
        
        '''self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.dim_head))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.dim_head))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)'''

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x (tensor): Input features with shape (B, H /patch_szie* W/patch_szie* D/patch_szie, hidden_size) or
            (B, H /patch_szie* W/patch_szie, hidden_size)

        Returns:
            tensor: Output features with same shape
        """
        output = self.input_rearrange(self.qkv(x)) #(B, N, 3*num_heads*C)->(qkv, B, num_heads, N, C)
        q, k, v = output[0], output[1], output[2] #(B, num_heads, N, C)
        if self.qk_normalization: ##!! maybe not right: need from (B, num_heads, N, C)->(B, N, num_heads*C)->(B, num_heads, N, C)
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale)
        '''if self.use_rel_pos:
            att_mat = add_decomposed_rel_pos(att_mat, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))'''
        att_mat = att_mat.softmax(dim=-1)
        
        if self.save_attn:
            # no gradients and new tensor;
            # https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
            self.att_mat = att_mat.detach()

        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.proj(x)
        x = self.drop_output(x)
        return x

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    Also, reference from ViTDet implement in https://github.com/facebookresearch/detectron2.git
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
        drop_path: float = 0.0,
        window_size: int = 0,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Sequence[int] | int | None = None,
        init_values: float|Tensor|None =None,
        attn_norm_layer: nn.Module = nn.LayerNorm,
        attn_qk_normalization: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            mlp_dim (int): dimension of feedforward layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
            drop_path (float, optional): Stochastic depth rate, fraction of the input samples to drop.,
            window_size (int, optional): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool, optional): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool, optional): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size and reshape.
            init_values=None, init value for LayerScale
            attn_norm_layer: nn.Module = nn.LayerNorm, norm for attn
            attn_qk_normalization: bool = False,, norm for attn q,k
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, save_attn,
                            use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                            input_size=input_size if window_size == 0 else (window_size, window_size),
                            norm_layer=attn_norm_layer,
                            qk_normalization=attn_qk_normalization)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ls1 = LayerScale(hidden_size, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(hidden_size, init_values=init_values) if init_values else nn.Identity()
        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (tensor): Input tensor with shape (B, H /patch_szie* W/patch_szie* D/patch_szie, hidden_size) or
            (B, H /patch_szie* W/patch_szie, hidden_size)

        Returns:
            tensor: Same shape of input
        """
        short_cut = x
        x = self.norm1(x)
        if self.window_size > 0: # (B, HW, C) -> (B, H, W, C) -> (B*windows, window_szie, window_size, C) -> (B*windows, window_szie*window_size, C)
            x = x.view(-1, self.input_size[0], self.input_size[1], self.hidden_size)
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            x = x.view(-1, self.window_size*self.window_size, self.hidden_size)
            x = self.attn(x) # same shape
            #(B*windows, window_szie*window_size, C)->(B*windows, window_szie, window_size, C)->(B, H, W, C)->(B, HW, C)
            x = x.view(-1, self.window_size, self.window_size, self.hidden_size)
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            x = x.view(-1, self.input_size[0]* self.input_size[1], self.hidden_size)
        else:
            x = self.attn(x) # same shape
        x = short_cut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

class ViT_new(nn.Module):
    def __init__(self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        drop_path_rate: float = 0.0,
        window_size: int = 0,
        window_block_indexes: list = [],
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        pretrain_img_size: int = 224,
        out_feature: str ="last_feat",
        init_values: float|Tensor|None =None,
        attn_norm_layer: nn.Module = nn.LayerNorm,
        attn_qk_normalization: bool = False,
        ):
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.
            drop_path_rate (float): Stochastic depth rate.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            pretrain_img_size (int): input image size for pretraining models.
            out_feature (str): name of the feature from the last block.
            init_values=None, init value for LayerScale
            attn_norm_layer: nn.Module = nn.LayerNorm,
            attn_qk_normalization: bool = False,
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        #add for clearify patch and image size/shape
        if isinstance(patch_size,int):
            patch_size_val = patch_size
            patch_shape = [patch_size] * spatial_dims
        else:
            assert len(patch_size) == spatial_dims
            patch_size_val = patch_size[0]
            patch_shape = patch_size
        
        if isinstance(img_size,int):
            img_size_val = img_size
            img_shape = [img_size] * spatial_dims
        else:
            assert len(img_size) == spatial_dims
            img_size_val = img_size[0]
            img_shape = img_size
        patched_input_shape = [int(img/patch) for img,patch in zip(img_shape,patch_shape)]
        self.patched_input_shape = patched_input_shape
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self._out_feature_channels = {out_feature: hidden_size}
        self._out_feature_strides = {out_feature: patch_size_val}
        self._out_features = [out_feature]
        self.hidden_size = hidden_size
        self.spatial_dims = spatial_dims

        self.classification = classification
        #position encoding
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_shape, #use shape for convenience
            patch_size=patch_shape,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        #blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn,
                                drop_path=dpr[i], 
                                window_size=window_size if i in window_block_indexes else 0,
                                use_rel_pos=use_rel_pos,
                                rel_pos_zero_init=rel_pos_zero_init,
                                input_size=patched_input_shape,
                                init_values=init_values,
                                attn_norm_layer=attn_norm_layer,
                                attn_qk_normalization=attn_qk_normalization,
                                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x):
        """forward
        Args:
            x (tensor): Input image with shape (B, C, H, W, D) or (B, C, H, W)

        Returns:
            last_features, (B, H/patch * W/patch, C)
            hidden layer's features, (B, H/patch * W/patch, C)
        """
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out
    
    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
