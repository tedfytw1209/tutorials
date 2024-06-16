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


def load_mednist_datalist(
    data_list_key: str, #"training" or "validation"
    base_dir: str, #dataset path
):
    root_dir = tempfile.mkdtemp() if base_dir is None else base_dir
    print(root_dir)
    data = MedNISTDataset(root_dir=root_dir, section=data_list_key, download=True, seed=0)
    datalist = [{"image": item["image"]} for item in data.data if item["class_name"] == "HeadCT"]
    return datalist
    
    