import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gc


class FP(nn.Module):
    def __init__(self, args):
        super(FP, self).__init__()

        self.args = args
        # angle setting
        self.rot_deg = 360
        self.d_beta = np.pi * self.rot_deg / self.args.view / 180  # angular step size in radian
        self.beta = -torch.linspace(0, (self.args.view - 1) * self.d_beta, self.args.view)
        self.args.DCD = self.args.SDD - self.args.SCD

        # Detector space, real, mm
        self.shift = 0
        if self.args.quarter_offset:
            self.shift = 0.25 * self.args.det_interval

        self.range_det = (
            torch.tensor([-self.args.det_interval * (self.args.num_det - 1) / 2, self.args.det_interval * (self.args.num_det - 1) / 2])
            + self.shift
        )

        self.param_setting()
        self.grid_pos, self.weighting = self.prepare(self.grid_pos, self.weighting)

    def param_setting(self):
        # Grid for Projection
        axis_x = (
            torch.linspace(-self.args.img_size[0] / 2, self.args.img_size[0] / 2, self.args.img_size[0] + 1) * self.args.pixel_size
        )  # cm scale, shape : (size+1) (1D)
        axis_y = torch.linspace(-self.args.img_size[1] / 2, self.args.img_size[1] / 2, self.args.img_size[1] + 1) * self.args.pixel_size

        if self.args.mode == "equiangular":
            center = (self.args.num_det + 1) / 2 - self.shift / self.args.det_interval
            delta_gamma = self.args.det_interval / self.args.SDD
            gamma_vec = (torch.linspace(1, self.args.num_det, self.args.num_det) - center) * delta_gamma
            range_det_x = self.args.SDD * torch.sin(gamma_vec)  # (n_det)
            range_det_y = -(self.args.SDD * torch.cos(gamma_vec) - self.args.SCD)
        elif self.args.mode == "equally_spaced":
            range_det_y = -(torch.ones(self.args.num_det) * self.args.DCD)
            range_det_x = (
                torch.linspace(-(self.args.num_det - 1) / 2, (self.args.num_det - 1) / 2, self.args.num_det) * self.args.det_interval
                + self.shift
            )

        src_zero = torch.tensor([0, self.args.SCD])
        self.src_point = torch.cat(
            (
                src_zero[0] * torch.cos(self.beta)[None, :] - src_zero[1] * torch.sin(self.beta)[None, :],
                +src_zero[0] * torch.sin(self.beta)[None, :] + src_zero[1] * torch.cos(self.beta)[None, :],
            ),
            dim=0,
        )  # (2, view)

        self.det_x_rot = (
            torch.cos(self.beta)[:, None] * range_det_x[None, :] - torch.sin(self.beta)[:, None] * range_det_y[None, :]
        )  # (view, n_det)
        self.det_y_rot = (
            torch.sin(self.beta)[:, None] * range_det_x[None, :] + torch.cos(self.beta)[:, None] * range_det_y[None, :]
        )  # (view, n_det)

        ax = (axis_x[None, None, :] - self.src_point[0][:, None, None]) / (self.det_x_rot - self.src_point[0, :][:, None])[
            :, :, None
        ]  # (view, det, grid+1)
        ay = (axis_y[None, None, :] - self.src_point[1][:, None, None]) / (self.det_y_rot - self.src_point[1, :][:, None])[:, :, None]

        a_min = torch.maximum(torch.minimum(ax[:, :, 0], ax[:, :, -1]), torch.minimum(ay[:, :, 0], ay[:, :, -1]))  # (view, det)
        a_min[a_min < 0] = 0
        a_max = torch.minimum(torch.maximum(ax[:, :, 0], ax[:, :, -1]), torch.maximum(ay[:, :, 0], ay[:, :, -1]))  # (view, det)
        a_max[a_max > 1] = 1

        axy = torch.cat([ax, ay], dim=2)
        axy[torch.logical_or(axy > a_max[:, :, None], axy < a_min[:, :, None])] = float("nan")
        axy, _ = torch.sort(axy, dim=2)
        diff_axy = torch.diff(axy, n=1, dim=2)  # shape(view, det, x+y+1)
        diff_axy[torch.isnan(diff_axy)] = 0
        a_mid = axy[:, :, :-1] + diff_axy / 2  # shape(view, det, x+y+1)
        a_mid[torch.isnan(a_mid)] = 0
        s2d = torch.sqrt(
            torch.pow(self.det_x_rot - self.src_point[0][:, None], 2) + torch.pow(self.det_y_rot - self.src_point[1][:, None], 2)
        )  # (view, det)
        self.weighting = diff_axy[None, None, :, :, :] * s2d[None, None, :, :, None]  # (batch, ch, view, det, x+y+1)

        x_pos = (
            a_mid * (self.det_x_rot[:, :, None] - self.src_point[0, :][:, None, None]) + self.src_point[0, :][:, None, None]
        )  # (view, det, x+y+1)
        x_pos = x_pos / (self.args.pixel_size * (self.args.img_size[0]) / 2)  # normalize grid
        y_pos = a_mid * (self.det_y_rot[:, :, None] - self.src_point[1, :][:, None, None]) + self.src_point[1, :][:, None, None]
        y_pos = y_pos / (self.args.pixel_size * (self.args.img_size[1]) / 2)

        self.grid_pos = torch.cat((x_pos.unsqueeze(-1), y_pos.unsqueeze(-1)), dim=3)  # (view, det, x+y+1, 2)

    def forward(self, img):  # image is [batch, 1, X, Y] shape tensor
        self.batch, self.ch, _, _ = img.size()
        interp_grid = F.grid_sample(
            img,
            torch.tile(self.grid_pos[:, :, :, :].view(1, self.args.num_det * self.args.view, -1, 2), [self.batch, 1, 1, 1]),
            padding_mode="zeros",
            mode="bilinear",
            align_corners=False,
        )
        interp_grid = interp_grid.view(self.batch, self.ch, self.args.view, self.args.num_det, -1)  # (batch, ch, view, det, x+y+1)
        sinogram = torch.sum(self.weighting[:, :, :, :, :] * interp_grid, dim=-1)
        sinogram[torch.isnan(sinogram)] = 0
        return sinogram

    def prepare(self, *args):
        device = torch.device("cpu" if self.args.cpu else "cuda")

        def _prepare(tensor):
            return tensor.to(device)

        if len(args) <= 1:
            return args[0].to(device)
        else:
            return [_prepare(a) for a in args]
