import os
import time
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import shutil
import torch.fft as fft


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def compute_num_params(model, text=False):
    num_params = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if num_params >= 1e6:
            return "{:.1f}M".format(num_params / 1e6)
        else:
            return "{:.1f}K".format(num_params / 1e3)
    else:
        return num_params


class checkpoint:
    def __init__(self, config, load="", save="", test_only=False):
        self.ok = True
        self.train_log = torch.Tensor()
        self.val_log = torch.Tensor()

        now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if load == "":
            if not save:
                save = now
            self.dir = os.path.join("/experiment", save)
        else:
            self.dir = os.path.join("/experiment", load)
            if os.path.exists(self.dir) and not test_only:
                self.train_log = torch.load(self.get_path("loss_log.pt"))
                self.val_log = torch.load(self.get_path("rmse_log.pt"))

        print("experiment directory is {}".format(self.dir))
        if not test_only:
            os.makedirs(self.dir, exist_ok=True)
            os.makedirs(self.get_path("model"), exist_ok=True)
            os.makedirs(self.get_path("results"), exist_ok=True)
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
            print("Config saved successfully.")

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def add_train_log(self, log):
        self.train_log = torch.cat([self.train_log, log])

    def add_val_log(self, log):
        self.val_log = torch.cat([self.val_log, log])

    def save(self, trainer, epoch, is_best=False, n_gpus=1):
        if n_gpus > 1:
            trainer.model.module.save(self.get_path("model"), epoch, is_best=is_best)
        else:
            trainer.model.save(self.get_path("model"), epoch, is_best=is_best)
        trainer.optimizer.save(self.dir)
        torch.save(self.train_log, os.path.join(self.dir, "loss_log.pt"))
        torch.save(self.val_log, os.path.join(self.dir, "rmse_log.pt"))


def calc_psnr(test, ref):
    mse = ((test - ref) ** 2).mean([-2, -1])
    return 20 * torch.log10(ref.max() / torch.sqrt(mse)).cpu().item()


def calc_rmse(img1, img2):
    mse = ((img1 - img2) ** 2).mean([-2, -1])
    return torch.sqrt(mse).mean().cpu().item()


def normalize(img1, mean, std):
    img1 = (img1 - mean) / std
    return img1


def denormalize(img, mean, std):
    img1 = img * std + mean
    return img1


def make_optimizer(optim_spec, target):
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {"lr": optim_spec["lr"], "weight_decay": optim_spec["weight_decay"]}

    if optim_spec["name"] == "SGD":
        optimizer_class = optim.SGD
    elif optim_spec["name"] == "ADAM":
        optimizer_class = optim.Adam
    elif optim_spec["name"] == "RMSprop":
        optimizer_class = optim.RMSprop
    elif optim_spec["name"] == "RADAM":
        optimizer_class = optim.RAdam

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))

        def get_dir(self, dir_path):
            return os.path.join(dir_path, "optimizer.pt")

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    return optimizer


def ramp_filter(ct_spec, scale):
    g = torch.zeros([1, ct_spec["num_det"] // scale])
    g[:] = float("nan")

    delta = scale * ct_spec["det_interval"] / ct_spec["SDD"]
    g[:, 0] = 1 / (8 * delta**2)
    for n in range(1, ct_spec["num_det"] // scale):
        if n % 2 == 1:
            g[:, n] = -0.5 / (np.pi * np.sin(n * delta)) ** 2
        else:
            g[:, n] = 0

    g = torch.cat([torch.fliplr(g[:, 1:]), g], axis=1)
    return g


def make_coord(shape, padding, dim, device):
    if dim == 1:
        det_seqs = -1 + 1 / shape[1] + (2 / shape[1]) * (torch.arange(shape[1] + padding * 2, device=device).float() - padding)
        coord = torch.tile(det_seqs, (shape[0], 1))
    elif dim == 2:
        coord_seqs = []
        for i, n in enumerate(shape):
            r = 2 / (2 * n)
            seq = -1 + r + (2 * r) * torch.arange(n, device=device).float()
            coord_seqs.append(seq)
        coord = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    return coord


def BP_grid(ct_spec):
    d_beta = np.pi * 2 / ct_spec["view"]  # angular step size in radian
    beta = -1 * torch.linspace(0, (ct_spec["view"] - 1) * d_beta, ct_spec["view"])

    x = ct_spec["recon_interval"] * torch.linspace(
        (1 - ct_spec["recon_size"][0]) / 2, (ct_spec["recon_size"][0] - 1) / 2, ct_spec["recon_size"][0]
    )
    y = ct_spec["recon_interval"] * torch.linspace(
        (1 - ct_spec["recon_size"][1]) / 2, (ct_spec["recon_size"][1] - 1) / 2, ct_spec["recon_size"][1]
    )
    x_mat, y_mat = torch.meshgrid(x, y, indexing="xy")
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
    view_xy_n = torch.tile(view_coord[:, None], [1, ct_spec["recon_size"][0] * ct_spec["recon_size"][1]])
    grid = torch.stack([s_xy_n, view_xy_n], dim=2)  # (view,x*y,2)
    square_inv = 1 / torch.reshape(torch.pow(L, 2), [ct_spec["view"], -1]).unsqueeze(0)  # (1, view, x*y)
    return grid, square_inv
