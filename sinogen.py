import pydicom
import torch
from preprocessing import FP
from preprocessing.CT_option import args
import os
import matplotlib.pyplot as plt
import numpy as np
import utility
import time

batch = 4
device = torch.device("cpu" if args.cpu else "cuda")
u_water = 0.0192867
args.num_split = 1
args.view = 512
args.ID_patient = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"]
Nin = 1e6


def main():
    args.pixel_size = 0.6640625
    args.det_interval = 1.2858393
    FP_model = FP.FP(args)

    for id in args.patient_ID:
        path = os.path.join("/Mayo", id, "full_1mm")
        save_path = os.path.join("/Dataset", id, "sinogram", str(args.view) + "views")
        print("save path is {}".format(save_path))
        os.makedirs(save_path, exist_ok=True)
        filenames = os.listdir(path)
        for i in filenames:
            data = pydicom.dcmread(os.path.join(path, i))
            img = data.pixel_array * data.RescaleSlope
            img = torch.FloatTensor(img - 1024).unsqueeze(0).unsqueeze(0).to(device)
            img = img * u_water / 1000 + u_water
            sinogram = FP_model(img)
            sino_photon = Nin * torch.exp(-sinogram)
            sino_noise = torch.poisson(sino_photon)
            sino_noise[sino_noise == 0] = 1
            sinogram = sino_noise.squeeze().cpu().numpy()
            np.save(os.path.join(save_path, i[:-4] + ".npy"), sinogram)


if __name__ == "__main__":
    main()
