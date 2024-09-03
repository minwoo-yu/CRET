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
    return IMGSR(config, restorator=config['model']['encoder']['name'])

class IMGSR(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.model_spec = config['model']
        self.device = torch.device('cpu' if config['cpu'] else 'cuda')

        module = import_module('model.' + config['model']['restorator']['name'].lower())
        self.restorator = module.make_model(scale=config['dataset']['max_scale'], 
                                        upsampler=config['model']['restorator']['upsampler'], 
                                        sino_unfold=config['model']['restorator']['sino_unfold'])
        
        self.tail = nn.Conv2d(self.encoder.out_dim, 1, 1, padding=0)

    def back_projection(self, sinogram, grid, square_inv):
        recon_img = F.grid_sample(sinogram, grid,
                                    mode='bilinear', padding_mode='border', align_corners=False)
        recon_img = torch.sum(recon_img * square_inv * 1000 , dim=-2) #(batch, x*y)
        return recon_img

    def forward(self, sinogram, grid, square_inv, cell, scale, train):
        batch, ch, view, det = sinogram.shape
        # feature extraction phase
        recon_feat = self.back_projection(sinogram, grid, square_inv)
        out = self.restorator(recon_feat)
        return out