import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from monai.data import DataLoader, Dataset
from PIL import Image
from monai.config import KeysCollection, PathLike

def load_mednist_datalist(
    data_list_key: str, #"training" or "validation"
    base_dir: str, #dataset path
):
    root_dir = tempfile.mkdtemp() if base_dir is None else base_dir
    print(root_dir)
    data = MedNISTDataset(root_dir=root_dir, section=data_list_key, download=True, seed=0)
    datalist = [{"image": item["image"]} for item in data.data if item["class_name"] == "HeadCT"]
    return datalist

def load_brainTR_datalist(
    data_list_key: str, #"training" or "validation"
    base_dir: str, #dataset path
):
    root_dir = tempfile.mkdtemp() if base_dir is None else base_dir
    print(root_dir)
    data = MedNISTDataset(root_dir=root_dir, section=data_list_key, download=True, seed=0)
    datalist = [{"image": item["image"]} for item in data.data if item["class_name"] == "HeadCT"]
    return datalist

### for eyeq dataset !!! not impl
def load_eyeq_datalist(
    data_list_key: str = "training",
    base_dir: PathLike | None = None,):
    """Load image paths of eyeq challenge from dir
    
    Args:
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.jpg'},
            {'image': '/workspace/data/chest_31.jpg'}
        ]

    """
    if data_list_key=="training":
        data_list = "train"
    elif data_list_key=="validation":
        data_list = "test"
    path = os.path.join(base_dir,data_list)
    a_images = os.listdir(path)

    expected_data = [{"image": os.path.join(path,fname)} for fname in a_images]

    return expected_data

class brainTR(Dataset):
    def __init__(self, root_a):
        self.root_a = root_a
        self.a_images = os.listdir(root_a)
        self.length_dataset = len(self.a_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        a_img = self.a_images[index % self.length_dataset]
        a_path = os.path.join(self.root_a, a_img)
        a_img = np.array(Image.open(a_path).convert("RGB"))

        return a_img