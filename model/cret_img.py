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
    return CRET_img(
        config,
        encoder=config["model"]["encoder"]["name"],
        restorator=config["model"]["restorator"]["name"],
        ensembling=config["model"]["ensembling"],
    )


class CRET_img(nn.Module):
    def __init__(self, config, encoder=None, restorator=None, ensembling=True):
        super().__init__()
        self.model_spec = config["model"]
        self.patch_size = config["dataset"]["patch_size"]
        self.device = torch.device("cpu" if config["cpu"] else "cuda")
        self.ensembling = ensembling

        if encoder is not None:
            module = import_module("model.cret_sino")
            self.step1 = module.make_model(config)
        else:
            self.step1 = None

        module = import_module("model." + self.model_spec["restorator"]["name"].lower())
        self.restorator = module.make_model(
            scale=1,
            upsampler=self.model_spec["restorator"]["upsampler"],
            input_dim=self.model_spec["restorator"]["input_dim"],
            config=config,
        )

        if self.model_spec["transfer_path"] is not None:
            load_from = torch.load(self.model_spec["transfer_path"])
            self.step1.load_state_dict(load_from, strict=True)
            print("Transfer Enabled")
            for param in self.step1.parameters():
                param.requires_grad = False

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
        for shift in shift_list:
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
        if self.step1 is not None:
            feat = self.step1.encoder(sinogram)
            recon_img = self.step1.querying(feat, grid, square_inv, scale).view(batch, 1, self.patch_size, self.patch_size)
            recon_img += self.back_projection(sinogram[:, ch // 2, :, :].unsqueeze(1), grid, square_inv).view(
                batch, 1, self.patch_size, self.patch_size
            )
        else:
            recon_img = self.back_projection(sinogram, grid, square_inv).view(batch, 1, self.patch_size, self.patch_size)
            # sino_unfold is not adapted for the image-domain only methods
        recon_img = self.restorator(recon_img)

        return recon_img
