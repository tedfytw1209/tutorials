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

    def __init__(self, image_keys: KeysCollection,box_keys: str,label_key: str,image_meta_key_postfix: str="meta_dict", allow_missing_keys: bool = False, training = True):
        self.image_keys = image_keys
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        self.box_keys = box_keys
        self.label_key = label_key
        box_ref_image_keys = image_keys[0]
        self.image_meta_key = f"{box_ref_image_keys}_{image_meta_key_postfix}"
        self.training = training

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        #box z, !! only select first box's mean
        box_arr = d[self.box_keys].cpu().detach().numpy()
        if box_arr.shape[0]>1: #multi boxs
            z_all_min = np.max(box_arr[:,2])
            z_all_max = np.min(box_arr[:,5])
            if z_all_max > z_all_min:
                z_center = int((z_all_max + z_all_min) / 2)
            else:
                z_center = int((box_arr[0,2] + box_arr[0,5]) / 2)
                select_boxs = []
                for i in range(box_arr.shape[0]):
                    if box_arr[i,2] <= z_center and box_arr[i,5] >= z_center:
                        select_boxs.append(i)
                d[self.box_keys] = torch.index_select(d[self.box_keys], 0, torch.LongTensor(select_boxs))#select box
                d[self.label_key] = torch.index_select(d[self.label_key], 0, torch.LongTensor(select_boxs))#select label
        elif box_arr.shape[0]==1: #one box
            z_min = box_arr[0,2]
            z_max = box_arr[0,5]
            z_center = int((z_min + z_max) / 2)
        else: #no box
            z_center = 0
        ### select med image in z domain and change shape
        image_key = self.image_keys[0]
        tmp = d[image_key]
        tmp = tmp[:,:,:,z_center]
        d[image_key] = tmp
        
        ### create new box value
        d['3d_box'] = torch.clone(d[self.box_keys])
        tmp_box = d[self.box_keys]
        tmp_box = torch.index_select(tmp_box, 1, torch.LongTensor([0, 1, 3, 4]))
        #tmp_box = tmp_box[:,:4]
        d[self.box_keys] = tmp_box
        
        ### aff change
        '''meta_dict = d[self.image_meta_key]
        affine = meta_dict["affine"]
        #print('3D Affine: ',affine.shape,affine)
        affine = torch.index_select(torch.index_select(affine, 1, torch.LongTensor([0, 1, 3])),0,torch.LongTensor([0, 1, 3]))
        #print('2D Affine: ',affine.shape,affine)
        meta_dict['affine'] = affine
        d[self.image_meta_key] = meta_dict'''
        
        '''print('Trans SelectTo2D Output:')
        for k in [self.image_keys[0],self.box_keys]:
            v = d[k]
            print(k, ": shape", v.shape)'''
        
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
            
            SelectTo2D(image_keys=[image_key], box_keys=box_key,label_key=label_key, training= True),
            EmptyBoxdTo2d(box_keys=[box_key], spatial_dims=2),
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
            SelectTo2D(image_keys=[image_key], box_keys=box_key, label_key=label_key, training=False),
            EmptyBoxdTo2d(box_keys=[box_key], spatial_dims=2),
            EnsureTyped(keys=[image_key, box_key], dtype=compute_dtype),
            EnsureTyped(keys=label_key, dtype=torch.long),
        ]
    )
    return val_transforms