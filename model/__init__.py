import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
from model import common
import numpy as np


class Model(nn.Module):
    def __init__(self, config, ckp):
        super(Model, self).__init__()
        print("Making model... {}".format(config["model"]["name"]))

        self.device = torch.device("cuda")
        module = import_module("model." + config["model"]["name"].lower())
        self.model = module.make_model(config).to(self.device)

        self.load(ckp.get_path("model"), resume=config["resume"])

    def forward(self, sinogram, grid, square_inv, scale):
        if self.training:
            return self.model(sinogram, grid, square_inv, scale)
        else:
            return self.model(sinogram, grid, square_inv, scale)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, "model_latest.pt")]
        if is_best:
            save_dirs.append(os.path.join(apath, "model_best.pt"))
        for i in save_dirs:
            torch.save(self.model.state_dict(), i)

    def load(self, apath, resume=-1):
        load_from = None
        kwargs = {}
        if resume == 0:
            print("Load the model from {}".format(os.path.join(apath, "model_latest.pt")))
            load_from = torch.load(os.path.join(apath, "model_latest.pt"), **kwargs)
        elif resume == 1:
            print("Load the model from {}".format(os.path.join(apath, "model_best.pt")))
            load_from = torch.load(os.path.join(apath, "model_best.pt"), **kwargs)
        elif resume == 2:
            load_from = torch.load(os.path.join(apath, "model_{}.pt".format(resume)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
