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

