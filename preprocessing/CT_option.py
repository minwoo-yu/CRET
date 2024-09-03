import argparse

parser = argparse.ArgumentParser(description="CT_geometry")

# Forward Projection
parser.add_argument("--img_size", type=list, default=[512, 512], help="Phantom image size")
parser.add_argument("--pixel_size", type=float, default=0.7421875, help="Pixel size of the phantom image")
parser.add_argument("--quarter_offset", action="store_true", help="detector quarter offset")
parser.add_argument("--geometry", type=str, default="fan", help="CT geometry")
parser.add_argument("--mode", type=str, default="equiangular", help="CT detector arrangement")
parser.add_argument("--view", type=int, default=512, help="number of view (should be even number for quarter-offset")
parser.add_argument("--num_split", type=int, default=512, help="number of splitting processes for FP")
parser.add_argument("--cpu", action="store_true", help="Activate CPU mode")

# Geometry conditions
parser.add_argument("--SCD", type=float, default=595, help="source-center distance (mm scale)")
parser.add_argument("--SDD", type=float, default=1085.6, help="source-detector distance (mm scale)")
parser.add_argument("--num_det", type=int, default=736, help="number of detector")
parser.add_argument("--det_interval", type=float, default=1.2858, help="interval of detector (mm scale)")
parser.add_argument("--det_lets", type=int, default=3, help="number of detector lets")

# Reconstruction
parser.add_argument("--window", type=str, default="rect", help="Reconstruction window")
parser.add_argument("--cutoff", type=float, default=0.3, help="Cutoff Frequency of some windows")
parser.add_argument("--ROIx", type=float, default=0, help="x ROI location")
parser.add_argument("--ROIy", type=float, default=0, help="y ROI location")
parser.add_argument("--recon_size", type=list, default=[512, 512], help="Reconstruction image size")
parser.add_argument("--recon_filter", type=str, default="ram-lak", help="Reconstruction Filter")
parser.add_argument("--recon_interval", type=float, default=0.7421875, help="Pixel size of the reconstruction image")
parser.add_argument("--num_interp", type=int, default=4, help="number of sinc interpolation in sinogram domain")
parser.add_argument("--no_mask", action="store_true", help="Not using Masking")

# Data directory
parser.add_argument("--data_dir", type=str, default="/Mayo", help="Dataset directory")
parser.add_argument(
    "--patient_ID", type=list, default=["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"], help="Patient ID"
)

# args = parser.parse_args()
args = parser.parse_args(args=[])

for arg in vars(args):
    if vars(args)[arg] == "True":
        vars(args)[arg] = True
    elif vars(args)[arg] == "False":
        vars(args)[arg] = False
    elif vars(args)[arg] == "None":
        vars(args)[arg] = None
