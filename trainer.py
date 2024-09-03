import os
import math
import utility
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import vessl
import numpy as np
import random
from model import common
from torch.optim.lr_scheduler import MultiStepLR
import vessl


class Trainer:
    def __init__(self, config, loader, my_model, ckp, load):
        self.config = config
        self.d_beta = np.pi * 2 / config["model"]["ct"]["view"]  # angular step size in radian
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loss = nn.L1Loss()
        self.model = my_model
        self.optimizer = utility.make_optimizer(config["optimizer"], self.model)
        self.device = torch.device("cuda")
        if load != "":
            self.optimizer.load(ckp.dir)
            # load at VESSL experiment
            for i in range(len(ckp.train_log)):
                vessl.log(step=i, payload={"train_loss": ckp.train_log[i].item()})
            for j in range(len(ckp.val_log)):
                vessl.log(step=j, payload={"val_rmse": ckp.val_log[j].item()})

        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config["optimizer"]["milestones"],
            gamma=config["optimizer"]["gamma"],
            last_epoch=len(ckp.train_log) - 1,
        )

        print("total number of parameter is {}".format(sum(p.numel() for p in self.model.parameters())))

        self.u_water = 0.0192867
        self.Nin = 1e6
        self.filtering = common.Filtering(config)
        self.squeezing = common.Block_sino(config)
        self.bp_grid, self.bp_square = utility.BP_grid(config["model"]["ct"])
        self.bp_grid, self.bp_square = self.prepare(self.bp_grid, self.bp_square)

        self.n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model)

    def train(self):
        epoch = self.scheduler.last_epoch
        self.ckp.add_train_log(torch.zeros(1))
        learning_rate = self.scheduler.get_last_lr()[0]
        self.model.train()
        train_loss = utility.Averager()
        timer = utility.Timer()
        for batch, (sino, img, mask_idx, patch_idx) in enumerate((self.loader_train)):
            sino, img = self.prepare(sino, img)
            scale = 2 ** random.randint(math.log2(self.config["dataset"]["min_scale"]), math.log2(self.config["dataset"]["max_scale"]))

            sino = F.avg_pool2d(sino, kernel_size=(1, scale), stride=(1, scale)) * scale  # detector binning
            sino = -torch.log(sino / self.Nin / scale)
            if self.config["dataset"]["squeezing"]:
                sino = F.pad(sino, (64 // scale, 64 // scale, 0, 0))

            sino = utility.normalize(sino, 0, 1)
            img = utility.normalize((img - 1024) * self.u_water / 1000 + self.u_water, 0, 1 * self.d_beta)

            sino = self.filtering(sino, scale)
            sino, bp_grid, bp_square = self.squeezing(sino, self.bp_grid, self.bp_square, mask_idx, patch_idx, scale)
            self.optimizer.zero_grad()
            recon_img = self.model(sino, bp_grid, bp_square, scale)
            loss = self.loss(recon_img, img)
            loss.backward()
            self.optimizer.step()
            train_loss.add(loss.item())
        vessl.log(step=epoch, payload={"train_loss": train_loss.item(), "train_time": timer.t(), "learning_rate": learning_rate})
        self.ckp.train_log[-1] = train_loss.item()
        self.scheduler.step()

    def eval(self):
        img_idx = torch.arange(512 * 512).reshape(512, 512)
        epoch = self.scheduler.last_epoch
        patch_size = self.config["dataset"]["patch_size"]
        sampling_size = self.config["dataset"]["sampling_size"]

        if epoch % self.config["test_every"] == 0:
            self.ckp.add_val_log(torch.zeros(1))
            self.model.eval()
            timer = utility.Timer()

            with torch.no_grad():
                for i, (sino, img, loc_name) in enumerate((self.loader_test)):
                    sino, img = self.prepare(sino, img)
                    batch, ch, h, w = sino.shape
                    if self.config["dataset"]["min_scale"] == self.config["dataset"]["max_scale"]:
                        scale = self.config["dataset"]["min_scale"]
                    else:
                        scale = int(2 ** (i // (len(self.loader_test) / self.config["dataset"]["valid"]["repeat"])))

                    sino = F.avg_pool2d(sino, kernel_size=(1, scale), stride=(1, scale)) * scale  # detector binning
                    sino = -torch.log(sino / self.Nin / scale)

                    if self.config["dataset"]["squeezing"]:
                        sino = F.pad(sino, (64 // scale, 64 // scale, 0, 0))

                    sino = utility.normalize(sino, 0, 1)

                    sino = self.filtering(sino, scale)

                    for x in range(0, 512, patch_size):
                        for y in range(128, 384, patch_size):
                            patch_idx = img_idx[y : y + patch_size, x : x + patch_size].flatten().reshape(1, -1).tile(batch, 1)
                            patch_img = img[:, :, y : y + patch_size, x : x + patch_size]
                            if sampling_size is not None:
                                idx = np.random.choice(patch_size**2, sampling_size**2, replace=False)
                                patch_img = patch_img.flatten(-2, -1)[:, :, idx].reshape(-1, 1, sampling_size, sampling_size)
                                patch_idx = patch_idx[:, idx]
                            mask_idx = [torch.tensor([y + 63]).tile(batch), torch.tensor([x + 63]).tile(batch)]
                            sino_patch, bp_grid, bp_square = self.squeezing(sino, self.bp_grid, self.bp_square, mask_idx, patch_idx, scale)

                            recon_img = self.model(sino_patch, bp_grid, bp_square, scale)  # (batch, x*y, ch)
                            recon_img = utility.denormalize(recon_img, 0, 1 * self.d_beta)
                            recon_img = ((recon_img - self.u_water) * 1000 / self.u_water).clamp_(-1024, 3071)  # mm-1 to HU
                            self.ckp.val_log[-1] += utility.calc_rmse(recon_img + 1024, patch_img) / len(self.loader_test) / 8

                best = self.ckp.val_log.min(0)  # best[0] is the minimum value, best[1] is the index of the minimum value
                vessl.log(step=epoch // self.config["test_every"] - 1, payload={"val_rmse": self.ckp.val_log[-1], "val_time": timer.t()})

            self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch // self.config["test_every"]), n_gpus=self.n_gpus)

    def prepare(self, *args):
        def _prepare(tensor):
            if tensor is not None:
                return tensor.to(self.device)
            else:
                return None

        return [_prepare(a) for a in args]

    def terminate(self):
        epoch = self.scheduler.last_epoch
        return epoch >= self.config["epochs"]
