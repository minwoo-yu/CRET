import math
import utility
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.autograd import Function
import torch.fft as fft


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor=2):
        super(PixelShuffle1D, self).__init__()
        self.scale = upscale_factor
        self.upscale = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_y, short_x = x.shape[-2:]

        long_channel_len = short_channel_len // self.upscale
        long_y = short_y
        long_x = self.upscale * short_x

        x = x.contiguous().view([batch_size, self.upscale, long_channel_len, short_y, short_x])
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # batch, long_channel_len, short_y, short_x, upscale
        x = x.view(batch_size, long_channel_len, long_y, long_x)
        return x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 2 * n_feats, 3, bias))
                m.append(PixelShuffle1D(2))  # detector dimension upsampling
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(inplace=True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 3 * n_feats, 3, bias))
            m.append(PixelShuffle1D(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Block_sino(nn.Module):
    def __init__(self, config):
        super(Block_sino, self).__init__()

        self.ct_spec = config["model"]["ct"]
        self.device = torch.device("cuda")

        self.center = torch.load(os.path.join("data/mask", "mask_info.pt")).to(self.device)
        self.sino_patch = 256  # 128*128 scale patch can be reconstructed via 256/scale detectors' info
        self.squeezing = config["dataset"]["squeezing"]
        self.unfolding = config["dataset"]["sino_unfold"]

    def forward(self, sinogram, bp_grid, bp_square, mask_idx, patch_idx, scale):
        batch, _, view, _ = sinogram.shape
        det = self.ct_spec["num_det"] // scale
        bp_grid = bp_grid[:, patch_idx, :].permute(1, 0, 2, 3)
        bp_square = bp_square[:, :, patch_idx].permute(2, 0, 1, 3)

        if self.unfolding:
            ch = 9
            sinogram = F.unfold(sinogram, kernel_size=3, padding=1).view(batch, ch, view, -1)
        else:
            ch = 1

        if self.squeezing:
            idx_det = (
                torch.arange(self.sino_patch // scale, device=self.device).view(1, -1, 1).tile(batch, 1, self.ct_spec["view"])
                + 64 // scale
                + self.center[mask_idx[0], mask_idx[1]].view(batch, 1, 512) // scale
                - (self.sino_patch // scale // 2)
            )
            idx_view = torch.arange(512, device=self.device).view(1, -1, 512).tile(batch, self.sino_patch // scale, 1)
            idx_batch = torch.arange(batch, device=self.device).view(-1, 1, 1).tile(1, self.sino_patch // scale, self.ct_spec["view"])

            sino_coord = utility.make_coord([view, det], padding=64 // scale, dim=1, device=self.device).tile(batch, 1, 1)  # (view, det)
            bp_minmax = torch.zeros((batch, self.ct_spec["view"], 2), device=self.device)
            sinogram = sinogram.permute(0, 2, 3, 1)[idx_batch, idx_view, idx_det].permute(0, 3, 2, 1)

            masked_bp = sino_coord[idx_batch, idx_view, idx_det].view(batch, -1, self.ct_spec["view"]).permute(0, 2, 1)
            bp_minmax = masked_bp[:, :, [0, -1]]
            bp_minmax[:, :, 0] += -1 / det
            bp_minmax[:, :, 1] += 1 / det
            bp_grid[:, :, :, 0] = (
                2
                / (bp_minmax[:, :, 1] - bp_minmax[:, :, 0]).view(batch, 512, 1)
                * (bp_grid[:, :, :, 0] - bp_minmax.mean(-1).view(batch, 512, 1))
            )

        return sinogram, bp_grid, bp_square


class Filtering(nn.Module):
    def __init__(self, config):
        super(Filtering, self).__init__()

        self.model_spec = config["model"]
        self.device = torch.device("cuda")
        # self.device = torch.device('cpu')
        delta_b = self.model_spec["ct"]["det_interval"] / self.model_spec["ct"]["SDD"]
        self.recon_filter = []

        for s in range(0, int(math.log2(config["dataset"]["max_scale"])) + 1):
            self.recon_filter.append(
                utility.ramp_filter(self.model_spec["ct"], 2**s)[None, None, :, :].to(self.device) * 2**s * delta_b
            )

    def forward(self, sinogram, scale):
        batch, ch, view, det = sinogram.shape

        # parameter initialization phase
        s_range = (
            1
            / self.model_spec["ct"]["SDD"]
            * (self.model_spec["ct"]["det_interval"] * torch.linspace((1 - det) / 2, (det - 1) / 2, det, device=self.device) * scale)
        )
        cos_weight = self.model_spec["ct"]["SCD"] / 10000 * torch.cos(s_range)
        # filtering phase
        sinogram = sinogram * cos_weight[None, None, None, :]
        sinogram = F.conv2d(
            sinogram,
            self.recon_filter[int(math.log2(scale))],
            padding=(0, int((self.model_spec["ct"]["num_det"] / scale * 2 - 1) // 2)),
        )
        return sinogram
