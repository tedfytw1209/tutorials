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

import math
import torch
from torch import Tensor, nn
from torch.nn import init
import torch.nn.functional as F

from timm.models.layers import DropPath

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.mlp import MLPBlock
from monai.utils import optional_import
from monai.networks.blocks.backbone_fpn_utils import BackboneWithFPN
from monai.networks.nets import ViT
from monai.networks.layers.factories import Conv, Pool
from monai.apps.detection.networks.retinanet_detector import *
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.data.box_utils import box_iou

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
        B, H, W, _ = x.shape
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
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
                            input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = nn.LayerNorm(hidden_size)
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
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViTDet(nn.Module):
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
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn,
                                drop_path=dpr[i], 
                                window_size=window_size if i in window_block_indexes else 0,
                                use_rel_pos=use_rel_pos,
                                rel_pos_zero_init=rel_pos_zero_init,
                                input_size=patched_input_shape,
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

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """forward

        Args:
            x (tensor): Input image with shape (B, C, H, W, D) or (B, C, H, W)

        Returns:
            Output last features: Dict with {self._out_features[0]: classification logits (B, num_classes) if self.classification==True
            or feature maps (B, C, H/patch, W/patch, D) or (B, C, H/patch, W/patch)}
        """
        #PatchEmbeddingBlock Input: (B, C, H, W, D), Output: (B, H /patch_szie* W/patch_szie* D/patch_szie, hidden_size)
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        #hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            #hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        # (B, H /patch_szie* W/patch_szie* D/patch_szie, hidden_size)->(B, H/patch * W/patch, C) ->(B, H/patch, W/patch, C)
        x = x.transpose(-1,-2).reshape(-1, self.patched_input_shape[0], self.patched_input_shape[1], self.hidden_size)
        return {self._out_features[0]:x}
    
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
        
class SimpleFeaturePyramid(nn.Module):
    """
    ViTDet, based on: "Yanghao Li et al.,
    Exploring plain vision transformer backbones for object detection <https://arxiv.org/abs/2203.16527>"
    Code come from https://github.com/facebookresearch/detectron2.git implement
    It creates pyramid features built on top of the input feature map.
    """
    def __init__(
        self,
        input_shapes: Sequence[int],
        in_feature: str,
        out_channels: int,
        scale_factors: Sequence[float],
        top_block: nn.Module | None = None,
        square_pad: int = 0,
        spatial_dims: int = 2,
    ):
        """
        Args:
            input_shapes: input_shapes of the net (net.output_shape()).
            in_feature (str): names of the input feature maps coming from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            square_pad (int): If > 0, require input images to be padded to specific square size.
            spatial_dims (int): default is 2, not implement spatial_dims=3 case now.
        """
        super().__init__()

        self.scale_factors = scale_factors
        self.spatial_dims = spatial_dims
        #input_shapes[in_feature].stride = patch_size, [4,8,16,32]
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        #_assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = nn.ModuleList() #for store sub modules
        use_bias = False
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 16.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2,eps=1e-5), #! detectron2 use 1e-6
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    LayerNorm(dim // 4,eps=1e-5), #! detectron2 use 1e-6
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                    LayerNorm(dim // 8,eps=1e-5), #! detectron2 use 1e-6
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 8, dim // 16, kernel_size=2, stride=2),
                ]
                out_dim = dim // 16
            elif scale == 8.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2,eps=1e-5), #!!! detectron2 use 1e-6
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    LayerNorm(dim // 4,eps=1e-5), #!!! detectron2 use 1e-6
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                ]
                out_dim = dim // 8
            elif scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2,eps=1e-5), #! detectron2 use 1e-6
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    nn.Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                    ),
                    LayerNorm(out_channels,eps=1e-5), #! detectron2 use 1e-6
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                    ),
                    #LayerNorm(out_channels,eps=1e-5), #! detectron2 use 1e-6 ##!!tmp remove it
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers) #torch func
            self.stages.append(layers)
        
        #self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"feat{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps, add another strides.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["feat{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Args:
            x: Output of vitdet model dict[str, Tensor(B, H/patch, W/patch, C)]
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["feat2", "feat3", ..., "feat6"]. patch default 16, 5 feature maps
                {
                    "feat2": Tensor(B,C, 4*H/patch, 4*W/patch)
                    "feat3": Tensor(B,C, 2*H/patch, 2*W/patch)
                    "feat4": Tensor(B,C, H/patch, W/patch)
                    "feat5": Tensor(B,C, H/(2*patch), W/(2*patch))
                    "feat6": Tensor(B,C, H/(4*patch), W/(4*patch))
                }
        """
        #bottom_up_features = self.net(x)
        bottom_up_features = x
        features = bottom_up_features[self.in_feature]
        features = features.permute(0,3,1,2)
        results: list[Tensor] = []

        for stage in self.stages:
            out_features = stage(features)
            results.append(out_features)

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.append(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        out_dict: dict[str, Tensor] = OrderedDict({f: res for f, res in zip(self._out_features, results)})
        return out_dict
    
class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """
    def __init__(self, spatial_dims: int = 2, in_feature: str = "feat2"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_feature
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        self.maxpool = pool_type(kernel_size=1, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool(x)

class BackboneWithFPN_vitdet(nn.Module):
    """
    Adds an FPN on top of a model. ViTdet version
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.

    Args:
        backbone: backbone network
        fpn: fpn module used
        return_layers: [not used] a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list: [not used] number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels: number of channels in the FPN.
        spatial_dims: 2D or 3D images
    """
    def __init__(
        self,
        backbone: nn.Module,
        fpn: nn.Module,
        return_layers: dict[str, str],
        in_channels_list: list[int],
        out_channels: int,
        spatial_dims: int | None = None,
    ) -> None:
        super().__init__()

        # if spatial_dims is not specified, try to find it from backbone.
        self.spatial_dims = spatial_dims
        self.model_spatial_dims = backbone.spatial_dims
        if self.spatial_dims!=self.model_spatial_dims:
            self.dim_change_flag = True
            print("Dim change from %d to %d"%(self.spatial_dims,self.model_spatial_dims))
        else:
            self.dim_change_flag = False

        #self.body = torchvision_models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers) #!!! not understand
        self.body = backbone
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Computes the resulted feature maps of the network.

        Args:
            x: input images

        Returns:
            feature maps after FPN layers. They are ordered from highest resolution first.
        """
        #change size if spatial_dim==2 and last dim==1
        #print('BackboneWithFPN_vitdet Input Features Shape: ', x.shape)
        if self.dim_change_flag and x.shape[-1]==1:
            x = torch.squeeze(x, dim=-1)
        features: dict[str, Tensor] = self.body(x)  # backbone
        y: dict[str, Tensor] = self.fpn(features)  # FPN
        if self.dim_change_flag: #change back for detector used
            out_dict: dict[str, Tensor] = {f: torch.unsqueeze(res,dim=-1) for f, res in y.items()}
        else:
            out_dict = y
        return out_dict
        

def _vit_fpn_extractor(
    backbone: nn.Module,
    fpn: nn.Module,
    spatial_dims: int,
    trainable_layers: int = 5, #! not used now
    returned_layers: list[int] | None = None,
) -> BackboneWithFPN_vitdet:
    """
    Same code as https://github.com/pytorch/vision/blob/release/0.12/torchvision/models/detection/backbone_utils.py
    Except that ``in_channels_stage2 = backbone.in_planes // 8`` instead of ``in_channels_stage2 = backbone.inplanes // 8``,
    and it requires spatial_dims: 2D or 3D images.
    """

    # select layers that wont be frozen
    #!! need ask about finetune process
    '''if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all(not name.startswith(layer) for layer in layers_to_train):
            parameter.requires_grad_(False)'''

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"feat{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [256 for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN_vitdet(
        backbone, fpn, return_layers, in_channels_list, out_channels, spatial_dims=spatial_dims
    )

#!! need check the pretrained backbone setting
def vitdet_fpn_feature_extractor(
    backbone: nn.Module,
    fpn: nn.Module,
    spatial_dims: int,
    pretrained_backbone: bool = False,
    returned_layers: Sequence[int] = (1, 2, 3),
    trainable_backbone_layers: int | None = None,
) -> BackboneWithFPN_vitdet:
    """
    Constructs a feature extractor network with a ViTdet-FPN backbone.
    Similar to resnet_fpn_feature_extractor in MONAI

    The returned feature_extractor network takes an image tensor as inputs,
    and outputs a dictionary that maps string to the extracted feature maps (Tensor).

    The input to the returned feature_extractor is expected to be a list of tensors,
    each of shape ``[C, H, W]`` or ``[C, H, W, D]``,
    one for each image. Different images can have different sizes.

    Args:
        backbone: a ResNet model, used as backbone.
        fpn: SimpleFeaturePyramid, used as fpn.
        spatial_dims: number of spatial dimensions of the images. We support both 2D and 3D images.
        pretrained_backbone: whether the backbone has been pre-trained.
        returned_layers: returned layers to extract feature maps. Each returned layer should be in the range [1,4].
            len(returned_layers)+1 will be the number of extracted feature maps.
            There is an extra maxpooling layer LastLevelMaxPool() appended.
        trainable_backbone_layers: number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
            When pretrained_backbone is False, this value is set to be 5.
            When pretrained_backbone is True, if ``None`` is passed (the default) this value is set to 3.
    """
    # If pretrained_backbone is False, valid_trainable_backbone_layers = 5.
    # If pretrained_backbone is True, valid_trainable_backbone_layers = trainable_backbone_layers or 3 if None.
    valid_trainable_backbone_layers: int = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, max_value=5, default_value=3
    )

    feature_extractor = _vit_fpn_extractor(
        backbone,
        fpn,
        spatial_dims,
        valid_trainable_backbone_layers,
        returned_layers=list(returned_layers)
    )
    return feature_extractor
