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
from monai.config.type_definitions import NdarrayOrTensor

from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms.transform import LazyTransform, MapTransform, Randomizable
from monai.transforms.inverse import InvertibleTransform
from monai.apps.detection.transforms.array import StandardizeEmptyBox
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Identityd,
    SqueezeDimd,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    ConvertBoxModed,
    StandardizeEmptyBoxd,
)


def generate_detection_train_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    patch_size,
    batch_size,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate training transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        box_key: the key to represent boxes in the input json files
        label_key: the key to represent box labels in the input json files
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        patch_size: cropped patch size for training
        batch_size: number of cropped patches from each image
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision

    Return:
        training transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    train_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            RandCropBoxByPosNegLabeld(
                image_keys=[image_key],
                box_keys=box_key,
                label_keys=label_key,
                spatial_size=patch_size,
                whole_box=True,
                num_samples=batch_size,
                pos=1,
                neg=1,
            ),
            RandZoomBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=2,
            ),
            RandRotateBox90d(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.75,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=[image_key, "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )
    return train_transforms


def generate_detection_val_transform(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        box_key: the key to represent boxes in the input json files
        label_key: the key to represent box labels in the input json files
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision
    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    val_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )
    return val_transforms


def generate_detection_inference_transform(
    image_key,
    pred_box_key,
    pred_label_key,
    pred_score_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
    spatial_dims=3,
):
    """
    Generate validation transform for detection.

    Args:
        image_key: the key to represent images in the input json files
        pred_box_key: the key to represent predicted boxes
        pred_label_key: the key to represent predicted box labels
        pred_score_key: the key to represent predicted classification scores
        gt_box_mode: ground truth box mode in the input json files
        intensity_transform: transform to scale image intensities,
            usually ScaleIntensityRanged for CT images, and NormalizeIntensityd for MR images.
        affine_lps_to_ras: Usually False.
            Set True only when the original images were read by itkreader with affine_lps_to_ras=True
        amp: whether to use half precision
        spatial_dims: dim of image(default 3), need SqueezeDim in spatial_dims=2.

    Return:
        validation transform for detection
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    ###!!! bug if spatial_dims==2
    if spatial_dims==3:
        add_transform = Identity()
    elif spatial_dims==2:
        add_transform = SqueezeDim(dim=-1)

    test_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key], dtype=torch.float32),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=compute_dtype),
        ]
    )
    post_transforms = Compose(
        [
            ClipBoxToImaged(
                box_keys=[pred_box_key],
                label_keys=[pred_label_key, pred_score_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys=[pred_box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertBoxModed(box_keys=[pred_box_key], src_mode="xyzxyz", dst_mode=gt_box_mode),
            DeleteItemsd(keys=[image_key]),
        ]
    )
    return test_transforms, post_transforms

### 2D version
class EmptyBoxdTo2d(MapTransform, InvertibleTransform):
    """
    When boxes are empty, this transform standardize it to shape of (0,4).
    """

    def __init__(self, box_keys: KeysCollection, spatial_dims: int=2, allow_missing_keys: bool = False) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            box_ref_image_keys: The single key that represents the reference image to which ``box_keys`` are attached.
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.ConvertBoxToStandardMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.spatial_dims = spatial_dims

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.converter = StandardizeEmptyBox(spatial_dims=self.spatial_dims)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        return dict(data)

class SelectTo2D(MapTransform):
    """
    Dictionary-based wrapper that select 2D slice from 3d imagse.
    image shape (C, H, W, D) e.g. (1, 540, 540, 247)
    box shape (N,6), [xmin, xmax, ymin, ymax, zmin, zmax] or [xmin, ymin, zmin, xmax, ymax, zmax] (standardized mode)

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        box_keys: box keys.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, image_keys: KeysCollection,box_keys: str,image_meta_key_postfix: str="meta_dict", allow_missing_keys: bool = False):
        self.image_keys = image_keys
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        self.box_keys = box_keys
        box_ref_image_keys = image_keys[0]
        self.image_meta_key = f"{box_ref_image_keys}_{image_meta_key_postfix}"

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        '''print('Trans SelectTo2D Input:')
        for k,v in d.items():
            print(k, ": ", v)'''
        ### !!! select the first image in z domain and change shape
        image_key = self.image_keys[0]
        tmp = d[image_key]
        tmp = tmp[:,:,:,0]
        d[image_key] = tmp
            
        ### create new box value
        tmp_box = d[self.box_keys]
        #tmp_box = torch.index_select(tmp_box, 1, torch.LongTensor([0, 1, 3, 4]))
        tmp_box = tmp_box[:,:4]
        d[self.box_keys] = tmp_box
        
        ### aff change
        meta_dict = d[self.image_meta_key]
        affine = meta_dict["affine"]
        print('2D Affine: ',affine.shape,affine)
        affine = torch.index_select(torch.index_select(affine, 1, torch.LongTensor([0, 1, 3])),0,torch.LongTensor([0, 1, 3]))
        print('2D Affine: ',affine.shape,affine)
        meta_dict['affine'] = affine
        
        '''print('Trans SelectTo2D Output:')
        for k,v in d.items():
            print(k, ": ", v)'''
        
        return d

def generate_detection_train_transform_2d(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    patch_size,
    batch_size,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate training transform for detection. (2D version)
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    
    spatial_dims=2
    

    train_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            #change
            SelectTo2D(image_keys=[image_key], box_keys=box_key),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            RandCropBoxByPosNegLabeld(
                image_keys=[image_key],
                box_keys=box_key,
                label_keys=label_key,
                spatial_size=patch_size,
                whole_box=True,
                num_samples=batch_size,
                pos=1,
                neg=1,
            ),
            RandZoomBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                padding_mode="constant",
                keep_size=True,
            ),
            ClipBoxToImaged(
                box_keys=box_key,
                label_keys=[label_key],
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipBoxd(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.5,
                spatial_axis=1,
            ),
            RandRotateBox90d(
                image_keys=[image_key],
                box_keys=[box_key],
                box_ref_image_keys=[image_key],
                prob=0.75,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            RandRotated(
                keys=[image_key, "box_mask"],
                mode=["nearest", "nearest"],
                prob=0.2,
                range_x=np.pi / 6,
                keep_size=True,
                padding_mode="zeros",
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            DeleteItemsd(keys=["box_mask"]),
            RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
            RandGaussianSmoothd(
                keys=[image_key],
                prob=0.1,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
            RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
            RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=[label_key], dtype=torch.long),
        ]
    )
    return train_transforms

def generate_detection_val_transform_2d(
    image_key,
    box_key,
    label_key,
    gt_box_mode,
    intensity_transform,
    affine_lps_to_ras=False,
    amp=True,
):
    """
    Generate validation transform for detection. (2D version)
    """
    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    val_transforms = Compose(
        [
            LoadImaged(keys=[image_key], image_only=False, meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.long),
            SelectTo2D(image_keys=[image_key], box_keys=box_key),
            StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
            Orientationd(keys=[image_key], axcodes="RAS"),
            intensity_transform,
            ConvertBoxToStandardModed(box_keys=[box_key], mode=gt_box_mode),
            AffineBoxToImageCoordinated(
                box_keys=[box_key],
                box_ref_image_keys=image_key,
                image_meta_key_postfix="meta_dict",
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )
    return val_transforms