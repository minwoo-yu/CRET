import torch
import utility
import data
import model
import os
import yaml
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from model import common
import torch.nn as nn
import vessl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test/test_rdn")
    parser.add_argument("--load", type=str, default="rdn_sino-x1")
    parser.add_argument("--scale", type=list, default=[1, 2, 4, 8])
    args = parser.parse_args()

    with open(os.path.join("configs", args.config + ".yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    vessl.init(message=args.config)

    u_water = 0.0192867
    d_beta = np.pi * 2 / config["model"]["ct"]["view"]  # angular step size in radian
    bp_grid, bp_square = utility.BP_grid(config["model"]["ct"])
    bp_grid, bp_square = bp_grid.cuda(), bp_square.cuda()

    filtering = common.Filtering(config)
    squeezing = common.Block_sino(config)

    checkpoint = utility.checkpoint(args.load, args.load, test_only=True)
    loader = data.Data(config, test_only=True)
    net = model.Model(config, checkpoint)
    net = torch.compile(net)
    net.eval()
    recon_3d = np.zeros((512, 512, len(loader.loader_test)), dtype=np.float32)
    with torch.no_grad():
        for s_i in range(len(args.scale)):
            scale = args.scale[s_i]
            psnr_val = 0
            for i, (sino, img, loc_name) in enumerate(tqdm(loader.loader_test)):
                sino = F.avg_pool2d(sino.cuda(), kernel_size=(1, scale), stride=(1, scale)) * scale  # detector binning
                sino = -torch.log(sino / 1e6 / scale)
                if config["dataset"]["squeezing"]:
                    sino = F.pad(sino, (64 // scale, 64 // scale, 0, 0))
                sino = utility.normalize(sino, 0, 1)
                sino = filtering(sino, scale)
                img_idx = np.arange(512 * 512).reshape(512, 512)
                recon_img = torch.zeros_like(img)
                with torch.no_grad():
                    for x in range(0, 512, 128):
                        for y in range(0, 512, 128):
                            patch_idx = img_idx[x : x + 128, y : y + 128].flatten().reshape(1, -1)
                            sino_patch, grid_patch, square_patch = squeezing(sino, bp_grid, bp_square, [x + 63, y + 63], patch_idx, scale)
                            recon = net(sino_patch, grid_patch, square_patch, scale)
                            recon_img[:, :, x : x + 128, y : y + 128] = recon * d_beta
                recon_img = (recon_img - u_water) / u_water * 1000
                recon_img = recon_img.clamp_(-1024, 3071) + 1024
                psnr_val += utility.calc_psnr(recon_img, img) / len(loader.loader_test)
            print("model: {}, scale: {}, psnr: {}".format(args.load, scale, psnr_val))


if __name__ == "__main__":
    main()
