import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility
import math
import os


def BP_grid_patch(ct_spec, mask_idx, patch_size, up_scale):
    d_beta = np.pi * 2 / ct_spec["view"]  # angular step size in radian
    beta = -1 * torch.linspace(0, (ct_spec["view"] - 1) * d_beta, ct_spec["view"])
    y_ind, x_ind = mask_idx

    x = (
        ct_spec["recon_interval"]
        / up_scale
        * torch.linspace(
            (1 - int(ct_spec["recon_size"][0] * up_scale)) / 2,
            (int(ct_spec["recon_size"][0] * up_scale) - 1) / 2,
            int(ct_spec["recon_size"][0] * up_scale),
        )
    )
    y = (
        ct_spec["recon_interval"]
        / up_scale
        * torch.linspace(
            (1 - int(ct_spec["recon_size"][1] * up_scale)) / 2,
            (int(ct_spec["recon_size"][1] * up_scale) - 1) / 2,
            int(ct_spec["recon_size"][1] * up_scale),
        )
    )
    x_mat, y_mat = torch.meshgrid(x, y, indexing="xy")
    x_mat = x_mat[
        (int(y_ind * up_scale) - patch_size // 2 + 1) : (int(y_ind * up_scale) + patch_size // 2 + 1),
        (int(x_ind * up_scale) - patch_size // 2 + 1) : (int(x_ind * up_scale) + patch_size // 2 + 1),
    ]
    y_mat = y_mat[
        (int(y_ind * up_scale) - patch_size // 2 + 1) : (int(y_ind * up_scale) + patch_size // 2 + 1),
        (int(x_ind * up_scale) - patch_size // 2 + 1) : (int(x_ind * up_scale) + patch_size // 2 + 1),
    ]

    r = torch.sqrt(torch.pow(x_mat, 2) + torch.pow(y_mat, 2))  # (x,y,2)
    phi = torch.atan2(y_mat, x_mat)  # (x,y)
    phi[torch.isnan(phi)] = 0
    L = torch.sqrt(
        torch.pow(ct_spec["SCD"] + r[None, :, :] * torch.sin(beta[:, None, None] - phi[None, :, :]), 2)
        + torch.pow(-r[None, :, :] * torch.cos(beta[:, None, None] - phi[None, :, :]), 2)
    )
    s_xy = torch.atan2(
        r[None, :, :] * torch.cos(beta[:, None, None] - phi[None, :, :]),
        ct_spec["SCD"] + r * torch.sin(beta[:, None, None] - phi[None, :, :]),
    )  # s value of each coord (view,x,y)
    s_xy_n = torch.reshape(s_xy, [ct_spec["view"], -1]) / (
        1 / ct_spec["SDD"] * ct_spec["det_interval"] * ct_spec["num_det"] / 2
    )  # normalize range as [-1 1], (view,x*y)
    view_coord = -1 + 1 / ct_spec["view"] + (2 / ct_spec["view"]) * torch.arange(ct_spec["view"]).float()
    view_xy_n = torch.tile(view_coord[:, None], [1, patch_size * patch_size])  # (view,x*y)
    grid_patch = torch.stack([s_xy_n, view_xy_n], dim=2)  # (view,x*y,2)
    square_patch = 1 / torch.reshape(torch.pow(L, 2), [ct_spec["view"], -1]).unsqueeze(0)  # (1, view, x*y)
    return grid_patch, square_patch


class sino_patching(nn.Module):
    def __init__(self, config):
        super(sino_patching, self).__init__()
        self.ct_spec = config["model"]["ct"]
        self.device = torch.device("cuda")
        self.center = torch.load(os.path.join("data/mask", "mask_info.pt")).to(self.device)
        self.sino_patch = 256  # 128*128 scale patch can be reconstructed via 256/scale detectors' info
        self.squeezing = config["dataset"]["squeezing"]
        self.unfolding = config["dataset"]["sino_unfold"]

    def forward(self, sinogram, bp_grid, mask_idx, scale):
        batch, _, view, _ = sinogram.shape
        det = self.ct_spec["num_det"] // scale
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
        return sinogram, bp_grid
