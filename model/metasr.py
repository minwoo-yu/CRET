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
    return MetaSR(config, encoder=config["model"]["encoder"]["name"], restorator=config["model"]["restorator"]["name"])


class MetaSR(nn.Module):
    def __init__(self, config, encoder=None, restorator=None):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_spec = config["model"]
        self.patch_size = (
            config["dataset"]["sampling_size"] if config["dataset"]["sampling_size"] is not None else config["dataset"]["patch_size"]
        )
        module = import_module("model." + config["model"]["encoder"]["name"].lower())
        self.encoder = module.make_model(
            scale=config["dataset"]["max_scale"],
            upsampler=self.model_spec["encoder"]["upsampler"],
            input_dim=self.model_spec["encoder"]["input_dim"],
            config=config,
        )
        imnet_spec = {"name": "mlp", "args": {"in_dim": 2, "out_dim": self.encoder.out_dim * 3, "hidden_list": [64]}}

        module = import_module("model." + imnet_spec["name"].lower())
        self.imnet = module.make_model(imnet_spec["args"]["in_dim"], imnet_spec["args"]["hidden_list"], imnet_spec["args"]["out_dim"])

        if restorator is not None:
            module = import_module("model." + self.model_spec["restorator"]["name"].lower())
            self.restorator = module.make_model(
                scale=1, upsampler=config["model"]["restorator"]["upsampler"], sino_unfold=config["model"]["restorator"]["sino_unfold"]
            )
        else:
            self.restorator = None

    def decoder(self, sinogram, grid, square_inv, cell, scale):
        batch, ch, view, det = sinogram.shape

        sinogram = F.unfold(sinogram, kernel_size=(1, 3), padding=(0, 1)).view(batch, ch * 3, view, det)

        sino_coord = torch.tile(
            utility.make_coord(sinogram.shape[-2:], padding=0, dim=1, device=self.device),
            dims=(batch, 1, 1, 1),
        )
        sino_coord -= (2 / sinogram.shape[-1]) / 2

        grid_ = grid.clone()
        grid_[:, :, :, 0] -= cell[:, :, :, 0] / 2
        grid_q = (grid_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)

        # (batch, view, x*y, ch)
        q_feat = F.grid_sample(sinogram, grid_q, mode="nearest", padding_mode="border", align_corners=False).permute(
            0, 2, 3, 1
        )  # (batch, view, x*y, ch)
        q_coord = F.grid_sample(sino_coord, grid_q, mode="nearest", padding_mode="border", align_corners=False).permute(
            0, 2, 3, 1
        )  # (batch, view, x*y, 2)

        rel_coord = (grid_[:, :, :, 0].unsqueeze(-1) - q_coord) * sinogram.shape[-1] / 2  # (batch, view, x*y, 1)
        r_rev = cell * det / 2
        inp = torch.cat([rel_coord, r_rev], dim=-1)

        query = q_coord.shape[2]
        pred = self.imnet(inp).view(batch * view * query, sinogram.shape[1], 1)
        pred = torch.bmm(q_feat.contiguous().view(batch * view * query, 1, -1), pred).view(batch, view, query, 1)
        return torch.sum(pred.permute(0, 3, 1, 2) * square_inv * 10000, dim=-2)

    def forward(self, sinogram, grid, square_inv, scale):
        # feature extraction phase
        batch, ch, _, det = sinogram.shape
        cell = 2 * torch.ones_like(grid[:, :, :, 0].unsqueeze(-1), device="cuda") / 256
        sinogram = self.encoder(sinogram)
        recon_img = self.decoder(sinogram, grid, square_inv, cell, scale)
        return recon_img.view(batch, 1, self.patch_size, self.patch_size)
