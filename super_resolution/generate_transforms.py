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

import torch
import numpy as np
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Dict
from monai.config.type_definitions import NdarrayOrTensor
from monai import transforms
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable
from monai.transforms import Identityd
from torchvision.transforms.functional import rgb_to_grayscale

from torcheval.metrics.image import PeakSignalNoiseRatio
from torcheval.metrics.image.ssim import StructuralSimilarity
from skimage.metrics import structural_similarity
from typing import Iterable, Optional, TypeVar
TStructuralSimilarity = TypeVar("TStructuralSimilarity")

class StructuralSimilarity_gray(StructuralSimilarity):
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
    def __init__(self, data_range, device):
        self.psnr = PeakSignalNoiseRatio(data_range=data_range, device=device)

    def __call__(self, outputs, targets):
        self.psnr.update(outputs, targets)
        return self.psnr.compute()
    
class SSIM():
    def __init__(self, device):
        self.ssim = StructuralSimilarity_gray(device=device)

    def __call__(self, outputs, targets):
        self.ssim.update(outputs, targets)
        return self.psnr.compute()

class ToGrayScale(MapTransform):
    """
    Dictionary-based wrapper that turn RGB image to gray scale
    image shape (C, H, W) e.g. (1, 540, 540)

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        self.keys = keys
        super().__init__(self, keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        image_key = self.keys[0]
        img = d[image_key]
        img = rgb_to_grayscale(img)
        d[image_key] = img
        
        return d

def generate_mednist_train_transforms(image_size=64, lowres_img_size=16, to_gray=False):
    if to_gray:
        add_process = ToGrayScale
    else:
        add_process = Identityd
    train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        add_process(keys=["image"]),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[image_size, image_size],
            padding_mode="zeros",
            prob=0.5,
        ),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        transforms.Resized(keys=["low_res_image"], spatial_size=(lowres_img_size, lowres_img_size)),
    ]
    )
    return train_transforms

def generate_mednist_validation_transforms(image_size=64, lowres_img_size=16, to_gray=False):
    if to_gray:
        add_process = ToGrayScale
    else:
        add_process = Identityd
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            add_process(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.Resized(keys=["image"], spatial_size=(image_size, image_size)),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(keys=["low_res_image"], spatial_size=(lowres_img_size, lowres_img_size)),
        ]
    )
    return val_transforms

