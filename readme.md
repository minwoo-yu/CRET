# CRET: Continuous Representation-based Reconstruction for Computed Tomography

This is the official implement of the paper "CRET: Continuous Representation-based Reconstruction for Computed Tomography"

## Environment
- python3
- pytorch 2.1.2+cuda11.8
- matplotlib, yaml, tqdm, timm, etc..

## Dataset
- Download AAPM-Mayo dataset at [here](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/)
- Our CRET requires sinogram data. To generate sinogram data, run forward projection as:
```commandline
python sinogen.py 
```
- After generating sinogram and ground truth image as '.npy' extension, put these datasets as folows:
```
/Dataset
├── L067
│     ├── img
│     ├── sinogram
├── L096
│     ├── img
│     ├── sinogram
...
```

## For training
training the reconstruction module corresponding to the step-1 & encoder as RDN (use 2 GPUs)
```
python main.py --config train/sino/train_rdn-cret --gpu 0,1 --save rdn_cret
```

training the restoration module corresponding to the step-2  (use 2 GPUs)
```
python main.py --config train/dual/train_rdn-cret+ --gpu 0,1 --save rdn_cret+
```

## Checkpoints
- Pre-trained modules and mask information can be downloaded in [here](https://drive.google.com/drive/folders/1_O0UYI2-17IxfMYvClGkDlHZgl-pdpNt?usp=drive_link) (Use an account for code releases for anonymization) 
- Note that mask_info.pt is information about the sinogram mask for sinogram squeezing and should be placed at "data/mask/"
## Demo
- We provide a demo trial of our proposed CRET in `demo.ipynb`

## Acknowledgements
Our code is built mainly with reference to [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch/tree/master) and [OPE-SR](https://github.com/gaochao-s/ope-sr/tree/main).  We appreciate them for their invaluable contributions to the development community.
