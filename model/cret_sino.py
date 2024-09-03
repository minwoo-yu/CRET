import os
import torch
import torch.nn as nn
import numpy as np
import utility
from importlib import import_module
from model import common
import torch.nn.functional as F
import math
from model import decoder


def make_model(config):
    return CRET_sino(
        config,
        encoder=config["model"]["encoder"]["name"],
        ensembling=config["model"]["ensembling"],
    )


class CRET_sino(nn.Module):
    def __init__(self, config, encoder=None, ensembling=False):
        super().__init__()
        self.model_spec = config["model"]
        self.patch_size = config["dataset"]["patch_size"]
        self.device = torch.device("cuda")
        self.ensembling = ensembling

        module = import_module("model." + self.model_spec["encoder"]["name"].lower())
        self.encoder = module.make_model(
            scale=config["dataset"]["max_scale"],
            upsampler=self.model_spec["encoder"]["upsampler"],
            input_dim=self.model_spec["encoder"]["input_dim"],
            config=config,
        )

        self.adapt_layer = nn.Conv2d(self.encoder.out_dim, 2 * self.model_spec["decoder"]["max_freq"] + 1, 3, padding=1)

        self.decoder = decoder.LC_decoder(max_freq=self.model_spec["decoder"]["max_freq"], omega=0.5 * math.pi)

    def back_projection(self, sinogram, grid, square_inv):
        recon_img = F.grid_sample(sinogram, grid, mode="bilinear", padding_mode="border", align_corners=False)
        recon_img = torch.sum(recon_img * square_inv * 10000, dim=-2)  # (batch, x*y)
        return recon_img

    def querying(self, sinogram, grid, square_inv, scale):
        sinogram = self.adapt_layer(sinogram)

        batch, ch, view, det = sinogram.shape
        # coordinate initialization phase
        sino_coord = torch.tile(
            utility.make_coord(sinogram.shape[-2:], padding=0, dim=1, device=self.device),
            dims=(batch, 1, 1, 1),
        )
        # (batch, 1, view, det)

        if self.ensembling:
            shift_interval = 1 / det
            eps = 1e-4
            shift_list = [
                torch.tensor([-shift_interval, 0], device=self.device),
                torch.tensor([shift_interval, 0], device=self.device),
            ]
        else:
            shift_list = [torch.tensor([0, 0], device=self.device)]
            eps = 0
        # ensemble back-projection phase
        recon_img = 0
        # appending (batch, view, x*y, 1)
        q_recons = []
        lengths = []
        for i, shift in enumerate(shift_list):  # rel coord 양, 음 순서
            grid_ = grid.clone() + shift[None, None, None, :] + 1e-6
            # (batch, view, x*y, ch)
            q_recon = F.grid_sample(
                sinogram,
                grid_,
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            q_coord = F.grid_sample(
                sino_coord,
                grid_,
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            rel_coord = (grid[:, :, :, 0].unsqueeze(-1) - q_coord) * det
            q_recons.append(self.decoder(q_recon, rel_coord))  # input: (batch, view, x*y, ch+1)

            lengths.append(torch.abs(rel_coord) + eps)
        tot_length = torch.stack(lengths).sum(dim=0)
        if self.ensembling:
            lengths[0], lengths[1] = lengths[1], lengths[0]

        for q_recon, length in zip(q_recons, lengths):
            recon_img += q_recon * length / tot_length

        return torch.sum(recon_img.permute(0, 3, 1, 2) * square_inv * 10000, dim=-2)

    def forward(self, sinogram, grid, square_inv, scale):
        # feature extraction phase
        batch, ch, _, _ = sinogram.shape
        feat = self.encoder(sinogram)
        # feat = torch.zeros((1, 64, 512, 256), device="cuda")
        recon_img = self.querying(feat, grid, square_inv, scale)
        recon_img += self.back_projection(sinogram[:, ch // 2, :, :].unsqueeze(1), grid, square_inv)
        return recon_img.view(batch, 1, self.patch_size, self.patch_size)
