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
    return LTE(
        config,
        encoder=config["model"]["encoder"]["name"],
        decoder=config["model"]["decoder"]["name"],
    )


class LTE(nn.Module):
    def __init__(self, config, encoder=None, decoder=None, decoder_in_dim=64):
        super().__init__()
        self.model_spec = config["model"]
        self.patch_size = (
            config["dataset"]["sampling_size"] if config["dataset"]["sampling_size"] is not None else config["dataset"]["patch_size"]
        )
        self.device = torch.device("cuda")

        module = import_module("model." + config["model"]["encoder"]["name"].lower())
        self.encoder = module.make_model(
            scale=config["dataset"]["max_scale"],
            upsampler=self.model_spec["encoder"]["upsampler"],
            input_dim=self.model_spec["encoder"]["input_dim"],
            config=config,
        )

        module = import_module("model." + self.model_spec["decoder"]["name"].lower())
        self.decoder = module.make_model(decoder_in_dim, self.model_spec["decoder"]["hidden_list"], 1)

        self.model_spec = config["model"]
        self.coef = nn.Conv2d(self.encoder.out_dim, decoder_in_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, decoder_in_dim, 3, padding=1)
        self.phase = nn.Linear(1, decoder_in_dim // 2, bias=False)

    def back_projection(self, sinogram, grid, square_inv):
        recon_img = F.grid_sample(sinogram, grid, mode="bilinear", padding_mode="border", align_corners=False)
        recon_img = torch.sum(recon_img * square_inv * 10000, dim=-2)  # (batch, x*y)
        return recon_img

    def querying(self, sinogram, grid, square_inv, cell, scale):
        batch, ch, view, det = sinogram.shape

        coeff = self.coef(sinogram)
        freqq = self.freq(sinogram)

        shift_interval = 1 / (sinogram.shape[-1])
        eps_shift = 1e-6
        shift_list = [
            torch.tensor([-(shift_interval + eps_shift), 0], device=self.device),
            torch.tensor([(shift_interval + eps_shift), 0], device=self.device),
        ]

        sino_coord = torch.tile(
            utility.make_coord(sinogram.shape[-2:], padding=0, dim=1, device=self.device),
            dims=(batch, 1, 1, 1),
        )
        # (batch, 1, view, det)

        preds = []
        lengths = []
        for shift in shift_list:
            # prepare coefficient & frequency
            # (batch, view, x*y, ch)
            q_coef = F.grid_sample(
                coeff,
                grid + shift[None, None, None, :],
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            q_freq = F.grid_sample(
                freqq,
                grid + shift[None, None, None, :],
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            q_coord = F.grid_sample(
                sino_coord,
                grid + shift[None, None, None, :],
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            rel_coord = (grid[:, :, :, 0].unsqueeze(-1) - q_coord) * sinogram.shape[-1]

            # basis generation
            q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
            q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
            q_freq = torch.sum(q_freq, dim=-2)
            q_freq += self.phase(cell)
            q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

            pred = self.decoder(torch.mul(q_coef, q_freq))
            preds.append(pred)

            lengths.append(torch.abs(rel_coord) + 1e-4)

        tot_length = torch.stack(lengths).sum(dim=0)
        lengths[0], lengths[1] = lengths[1], lengths[0]

        ret = 0
        for pred, length in zip(preds, lengths):
            ret += pred * length / tot_length

        return torch.sum(ret.permute(0, 3, 1, 2) * square_inv * 10000, dim=-2)

    def forward(self, sinogram, grid, square_inv, scale):
        batch, ch, _, _ = sinogram.shape
        cell = 2 * torch.ones_like(grid[:, :, :, 0].unsqueeze(-1), device="cuda") / scale
        # feature extraction phase
        res = self.back_projection(sinogram[:, ch // 2, :, :].unsqueeze(1), grid, square_inv)
        sinogram = self.encoder(sinogram)
        # sinogram = torch.zeros((1, 64, 512, 256), device="cuda")
        recon_img = res + self.querying(sinogram, grid, square_inv, cell, scale)
        return recon_img.view(batch, 1, self.patch_size, self.patch_size)
