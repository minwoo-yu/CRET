import os
import glob
from data import common
import numpy as np
import torch.utils.data as data
import skimage
from pathlib import Path
import torch
import pydicom
import time
import torch.nn.functional as F


class SRData(data.Dataset):
    def __init__(self, config, mode="train", patient_ID=None, augment=False, cache="none"):
        self.data_spec = config["dataset"]
        self.model_spec = config["model"]
        self.mode = mode
        self.patient_ID = patient_ID
        self.augment = augment
        self.cache = cache
        self.device = torch.device("cuda")
        self.names_sino, self.names_img = self._scan()

    def __getitem__(self, idx):
        sino, img, loc_name = self._load_file(idx)
        if self.mode == "train":
            sino, img, mask_idx, patch_idx = self.preparation(sino, img, self.data_spec, self.mode)
            return sino, img, mask_idx, patch_idx
        else:
            return np.expand_dims(sino, 0), np.expand_dims(img, 0), loc_name

    def __len__(self):
        if self.mode == "train":
            self.dataset_length = int(len(self.names_sino) * self.data_spec["train"]["repeat"])
        elif self.mode == "valid":
            self.dataset_length = int(len(self.names_sino) * self.data_spec["valid"]["repeat"])
        else:
            self.dataset_length = len(self.names_sino)
        return self.dataset_length

    def _scan(self):
        names_sino = []
        names_img = []
        for ID in self.patient_ID:
            data_path = os.path.join(self.data_spec["data_dir"], ID)
            names_sino += glob.glob(os.path.join(data_path, "sinogram", "512views", "*.npy"))
            names_img += glob.glob(os.path.join(data_path, "img", "*.npy"))
        return names_sino, names_img

    def _get_index(self, idx):
        return idx % len(self.names_sino)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        loc_name = self.names_img[idx]

        sinogram = np.load(self.names_sino[idx])
        img = np.load(self.names_img[idx])

        loc_name = Path(loc_name).parts[-3::]
        return sinogram, img, loc_name

    def preparation(self, sino, img, data_spec, mode):
        sino, img = sino.astype(np.float32), img.astype(np.float32)
        if self.augment and mode == "train":
            sino, img = common.augment(sino, img)
        img, mask_idx, patch_idx = common.get_patch(
            img,
            patch_size=data_spec["patch_size"],
            sampling_size=data_spec["sampling_size"],
        )
        sino, img = np.expand_dims(sino, 0), np.expand_dims(img, 0)
        return sino, img, mask_idx, patch_idx
