import os
import torch
import torch.nn as nn
import numpy as np
import utility
from importlib import import_module
from model import common
import torch.nn.functional as F
import math


def make_model(config):
    return LIIF(
        config,
        encoder=config["model"]["encoder"]["name"],
        decoder=config["model"]["decoder"]["name"],
        feat_unfold=config["model"]["feat_unfold"],
        ensembling=config["model"]["ensembling"],
        cell_decoding=config["model"]["cell_decoding"],
    )


class LIIF(nn.Module):
    def __init__(self, config, encoder=None, decoder=None, feat_unfold=False, ensembling=False, cell_decoding=False):
        super().__init__()
        self.model_spec = config["model"]
        self.patch_size = (
            config["dataset"]["sampling_size"] if config["dataset"]["sampling_size"] is not None else config["dataset"]["patch_size"]
        )
        self.device = torch.device("cuda")
        self.ensembling = ensembling
        self.cell_decoding = cell_decoding
        self.feat_unfold = feat_unfold

        module = import_module("model." + config["model"]["encoder"]["name"].lower())
        self.encoder = module.make_model(
            scale=config["dataset"]["max_scale"],
            upsampler=self.model_spec["encoder"]["upsampler"],
            input_dim=self.model_spec["encoder"]["input_dim"],
            config=config,
        )

        if decoder is not None:
            decoder_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                decoder_in_dim *= 3
            decoder_in_dim += 1  # attach coord
            if self.cell_decoding:
                decoder_in_dim += 1  # attach cell
            module = import_module("model." + self.model_spec["decoder"]["name"].lower())
            self.decoder = module.make_model(decoder_in_dim, self.model_spec["decoder"]["hidden_list"], 1)
        else:
            self.decoder = None

    def querying(self, sinogram, grid, square_inv, cell, scale):
        batch, ch, view, det = sinogram.shape

        if self.decoder is None:
            # back-projection phase
            recon_img = F.grid_sample(sinogram, grid, mode="bilinear", padding_mode="border", align_corners=False)
            recon_img = torch.sum(recon_img * square_inv * 10000, dim=-2)  # (batch, x*y)
            return recon_img

        if self.feat_unfold:
            sinogram = F.unfold(sinogram, kernel_size=(1, 3), padding=(0, 1)).view(batch, ch * 3, view, det)

        # coordinate initialization phase
        sino_coord = torch.tile(
            utility.make_coord(sinogram.shape[-2:], padding=0, dim=1, device=self.device),
            dims=(batch, 1, 1, 1),
        )
        # (batch, 1, view, det)
        if self.ensembling:
            shift_interval = 1 / (sinogram.shape[-1])
            eps_shift = 1e-6
            shift_list = [
                torch.tensor([-(shift_interval + eps_shift), 0], device=self.device),
                torch.tensor([(shift_interval + eps_shift), 0], device=self.device),
            ]
        else:
            shift_list = [torch.tensor([0, 0], device=self.device)]
            eps_shift = 0

        # ensemble back-projection phase
        recon_img = 0
        # appending (batch, view, x*y, 1)
        q_recons = []
        lengths = []
        for shift in shift_list:
            q_recon = F.grid_sample(
                sinogram, grid + shift[None, None, None, :], mode="nearest", padding_mode="border", align_corners=False
            ).permute(
                0, 2, 3, 1
            )  # (batch, view, x*y, ch)
            q_coord = F.grid_sample(
                sino_coord, grid + shift[None, None, None, :], mode="nearest", padding_mode="border", align_corners=False
            ).permute(0, 2, 3, 1)

            rel_coord = (grid[:, :, :, 0].unsqueeze(-1) - q_coord) * sinogram.shape[-1]

            if self.decoder is not None:
                q_recon = torch.cat((q_recon, rel_coord), dim=-1)
                if self.cell_decoding:
                    q_recon = torch.cat((q_recon, cell), dim=-1)
                q_recons.append(self.decoder(q_recon))  # input: (batch, view, x*y, ch+1)

            lengths.append(torch.abs(rel_coord) + 1e-4)
        tot_length = torch.stack(lengths).sum(dim=0)

        if self.ensembling:
            lengths[0], lengths[1] = lengths[1], lengths[0]

        for q_recon, length in zip(q_recons, lengths):
            recon_img += q_recon * length / tot_length

        return torch.sum(recon_img.permute(0, 3, 1, 2) * square_inv * 10000, dim=-2)

    def forward(self, sinogram, grid, square_inv, scale):
        # feature extraction phase
        batch, _, _, _ = sinogram.shape
        cell = 2 * torch.ones_like(grid[:, :, :, 0].unsqueeze(-1), device="cuda") / scale
        sinogram = self.encoder(sinogram)
        # sinogram = torch.zeros((1, 64, 512, 256), device="cuda")
        recon_img = self.querying(sinogram, grid, square_inv, cell, scale)
        recon_img = recon_img.view(batch, 1, self.patch_size, self.patch_size)
        return recon_img
