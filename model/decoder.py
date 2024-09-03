import os
import torch
import torch.nn as nn
import numpy as np
import utility
from importlib import import_module
from model import common
import torch.nn.functional as F
import math


def get_embed_fns(max_freq):
    embed_fns = []
    embed_fns.append(lambda x: torch.ones((x.shape)))  # x: bsize, view, recon_size, 1
    for i in range(1, max_freq + 1):
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x * freq))
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x * freq))
    return embed_fns


class OPE(nn.Module):
    def __init__(self, max_freq, omega):
        super(OPE, self).__init__()
        self.max_freq = max_freq
        self.omega = omega
        self.embed_fns = get_embed_fns(self.max_freq)

    def forward(self, coords):
        """
        N,bsize,1 ---> N,bsize,1,2n+1
        """
        # self.embed_fns: [1, cos, sin, cos2, sin2, cos3, sin3, ...]
        ope_out = torch.cat([fn(self.omega * coords).to(coords.device) for fn in self.embed_fns], -1)
        return ope_out


class LC_decoder(nn.Module):
    """
    linear combination of OPE with single channel data
    """

    def __init__(self, max_freq, omega):
        super(LC_decoder, self).__init__()
        self.max_freq = max_freq
        self.ope = OPE(max_freq=max_freq, omega=omega)

    def forward(self, latent, rel_coord):
        """
        N,bsize,ccc N,bsize,ccc ---> N,bsize,1
        """
        ope_out = self.ope(rel_coord).unsqueeze(-2)
        ans = torch.matmul(ope_out, latent.unsqueeze(-1)).squeeze(-1)
        return ans
