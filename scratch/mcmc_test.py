import os
import sys

os.environ["MJ_TODROOT"] = os.path.dirname(os.path.abspath(__file__)) + "/input"
os.environ["MJ_OUTROOT"] = os.path.dirname(os.path.abspath(__file__)) + "/output"

import corner
import matplotlib.pyplot as plt
import minkasi
import numpy as np

from witch import mapmaking as mm
from witch import sampler
from witch.containers import Model
from witch.fitter import get_outdir, load_config, load_tods, make_parser, process_tods

np.random.seed(6958476)

# ---------------------------------------------------------------------------

parser = make_parser()
args = parser.parse_args()

cfg = load_config({}, args.config)
cfg["fit"] = cfg.get("fit", "model" in cfg)
cfg["sim"] = cfg.get("sim", False)
cfg["map"] = cfg.get("map", True)
cfg["sub"] = cfg.get("sub", True)
if args.nosub:
    cfg["sub"] = False
if args.nofit:
    cfg["fit"] = False

todvec = load_tods(cfg)

lims = todvec.lims()
pixsize = np.deg2rad(cfg["coords"]["dr"] / 3.60e03)  # np.deg2rad(1.00/3600)
skymap = minkasi.maps.SkyMap(lims, pixsize)

model = Model.from_cfg(cfg)
params = np.array(model.pars)

gridmx, gridmy = np.meshgrid(np.arange(skymap.nx), np.arange(skymap.ny))
gridwx, gridwy = skymap.wcs.all_pix2world(gridmx, gridmy, 0)
gridwx = np.fliplr(gridwx)
gridwz = np.linspace(
    -1 * cfg["coords"]["r_map"],
    cfg["coords"]["r_map"],
    2 * int(cfg["coords"]["r_map"] / cfg["coords"]["dr"]),
    dtype=float,
)

xyz = [
    gridwx[0, :][..., None, None],
    gridwy[:, 0][None, ..., None],
    gridwz[None, None, ...],
    eval(cfg["coords"]["x0"]),
    eval(cfg["coords"]["y0"]),
]

model.xyz = xyz

noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(
    cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
)

# ---------------------------------------------------------------------------

outdir = get_outdir(cfg, bowl_str, model)

res = sampler.sample(model, todvec, skymap, nwalk=20, nstep=50, nburn=50)
samples = res.get_chain(thin=1, flat=True)

for p in range(samples.shape[1]):
    print(corner.quantile(samples[:, p], [0.16, 0.50, 0.84]))
