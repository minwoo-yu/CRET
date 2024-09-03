import numpy as np
import torch
import torch.nn.functional as F

# import torchvision.transforms.functional.crop as crop
import torch.nn as nn
import random
import os
import math
import utility
import time


def get_patch(img, patch_size, sampling_size):
    ih, iw = img.shape
    patch_idx = np.arange(512 * 512).reshape(512, 512)

    iy = random.randrange(0, ih - patch_size + 1)
    ix = random.randrange(0, iw - patch_size + 1)

    img_patch = img[iy : iy + patch_size, ix : ix + patch_size]
    patch_idx = patch_idx[iy : iy + patch_size, ix : ix + patch_size]

    mask_idx = [iy + patch_size // 2 - 1, ix + patch_size // 2 - 1]

    patch_idx = patch_idx.flatten()
    img_patch = img_patch.flatten()

    if sampling_size is not None:
        s_idx = np.random.choice(patch_size**2, sampling_size**2, replace=False)
        patch_idx = patch_idx[s_idx]
        img_patch = img_patch[s_idx]
        patch_size = sampling_size
        
    return img_patch.reshape(patch_size, patch_size), mask_idx, patch_idx


def augment(sino, img, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot * random.randint(0, 4)
    view, det = sino.shape

    if hflip:
        img = img[:, ::-1]
        sino = sino[:, ::-1]
        sino = np.concatenate((np.expand_dims(sino[0, :], axis=0), sino[:0:-1, :]), axis=0)
    if vflip:
        img = img[::-1, :]
        sino = sino[::-1, ::-1]
        sino = np.concatenate((sino[view // 2 - 1 :, :], sino[: view // 2 - 1, :]), axis=0)
    if rot90:
        img = np.rot90(img, -rot90)
        sino = np.concatenate((sino[view // 4 * rot90 : :, :], sino[0 : view // 4 * rot90, :]), axis=0)
    return sino, img
