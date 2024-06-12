"""
Master fitting and map making script.
Docs will exist someday...
"""

import argparse as argp
import glob
import os
import sys
import time
from copy import deepcopy

import minkasi
import numpy as np
import yaml

import minkasi.tools.presets_by_source as pbs

from . import core
from . import mapmaking as mm
from . import utils as mju
from .containers import Model


def print_once(*args):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Arguments:

        *args: Arguments to pass to print.
    """
    if minkasi.myrank == 0:
        print(*args)
        sys.stdout.flush()


def make_parser() -> argp.ArgumentParser:
    # Parse arguments
    parser = argp.ArgumentParser(
        description="Fit cluster profiles using mikasi and minkasi_jax"
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "--nofit",
        "-nf",
        action="store_true",
        help="Don't actually fit, just use values from config",
    )
    parser.add_argument(
        "--nosub",
        "-ns",
        action="store_true",
        help="Don't subtract the sim'd or fit model. Useful for mapmaking a sim'd cluster",
    )
    parser.add_argument(
        "--wnoise",
        "-wn",
        action="store_true",
        help="Use whitenoise instead of map noise. Only for use with sim",
    )
    return parser


def load_tods(cfg: dict) -> minkasi.tods.TodVec:
    todroot = cfg["paths"]["tods"]
    if not os.path.isabs(todroot):
        todroot = os.path.join(
            os.environ.get("MJ_TODROOT", os.environ["HOME"]), todroot
        )
    tod_names = glob.glob(os.path.join(todroot, cfg["paths"]["glob"]))
    bad_tod, _ = pbs.get_bad_tods(
        cfg["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
    )
    if "cut" in cfg["paths"]:
        bad_tod += cfg["paths"]["cut"]
    tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)
    tod_names.sort()
    ntods = cfg["minkasi"].get("ntods", None)
    tod_names = tod_names[:ntods]
    tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
    minkasi.barrier()  # Is this needed?

    todvec = minkasi.tods.TodVec()
    for fname in tod_names:
        dat = minkasi.tods.io.read_tod_from_fits(fname)
        minkasi.tods.processing.truncate_tod(dat)
        minkasi.tods.processing.downsample_tod(dat)
        minkasi.tods.processing.truncate_tod(dat)
        # figure out a guess at common mode and (assumed) linear detector drifts/offset
        # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
        dd, pred2, cm = minkasi.tods.processing.fit_cm_plus_poly(
            dat["dat_calib"], cm_ord=3, full_out=True
        )
        dat["dat_calib"] = dd
        dat["pred2"] = pred2
        dat["cm"] = cm

        tod = minkasi.tods.Tod(dat)
        todvec.add_tod(tod)

    return todvec


def process_tods(
    cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
) -> str:
    bowling = cfg.get("bowling", {})
    sub_poly = bowling.get("sub_poly", False)
    method = bowling.get("method", "pred2")
    degree = bowling.get("degree", 2)
    sim = cfg.get("sim", False)
    for i, tod in enumerate(todvec.tods):
        ipix = skymap.get_pix(tod)
        tod.info["ipix"] = ipix

        if sub_poly:
            tod.set_apix()
            for j in range(tod.info["dat_calib"].shape[0]):
                x, y = (
                    tod.info["apix"][j],
                    tod.info["dat_calib"][j] - tod.info[method][j],
                )
                res = np.polynomial.polynomial.polyfit(x, y, degree)
                tod.info["dat_calib"][j] -= np.polynomial.polynomial.polyval(x, res)

        if sim:
            if cfg["wnoise"]:
                temp = np.percentile(np.diff(tod.info["dat_calib"]), [33, 68])
                scale = (temp[1] - temp[0]) / np.sqrt(8)
                tod.info["dat_calib"] = np.random.normal(
                    0, scale, size=tod.info["dat_calib"].shape
                )
            else:
                tod.info["dat_calib"] *= (-1) ** (
                    (minkasi.myrank + minkasi.nproc * i) % 2
                )
            pred = core.model_tod(
                model.xyz,
                *model.n_struct,
                model.dz,
                model.beam,
                (tod.info["dx"] - model.x0) * mju.rad_to_arcsec,
                (tod.info["dy"] - model.y0) * mju.rad_to_arcsec,
                *model.pars,
            )
            tod.info["dat_calib"] += np.array(pred)

        tod.set_noise(noise_class, *noise_args, **noise_kwargs)
    if sub_poly:
        return f"-{method}_{degree}"
    return ""


def get_outdir(cfg, bowl_str, model):
    name = model.name + ("_ns" * (not cfg["sub"]))
    outroot = cfg["paths"]["outroot"]
    if not os.path.isabs(outroot):
        outroot = os.path.join(
            os.environ.get("MJ_OUTROOT", os.environ["HOME"]), outroot
        )
    outdir = os.path.join(outroot, cfg["name"], name)
    if "subdir" in cfg["paths"]:
        outdir = os.path.join(outdir, cfg["paths"]["subdir"])
    if cfg["fit"]:
        outdir = os.path.join(
            outdir,
            "-".join(
                [
                    name
                    for name, to_fit in zip(model.par_names, model.to_fit_ever)
                    if to_fit
                ]
            ),
        )
    else:
        outdir = os.path.join(outdir, "not_fit")
    outdir += bowl_str
    if cfg["sim"]:
        outdir += "-sim"

    print_once("Outputs can be found in", outdir)
    if minkasi.myrank == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

    return outdir


def deep_merge(a: dict, b: dict) -> dict:
    """
    Based on https://gist.github.com/angstwad/bf22d1822c38a92ec0a9?permalink_comment_id=3517209
    """
    result = deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = deepcopy(bv)
    return result


def load_config(start_cfg, cfg_path):
    """
    We want to load a config and if it has the key "base",
    load that as well and merge them.
    We only want to take things from base that are not in the original config
    so we merge the original into the newly loaded one.
    """
    with open(cfg_path, "r") as file:
        new_cfg = yaml.safe_load(file)
    cfg = deep_merge(new_cfg, start_cfg)
    if "base" in new_cfg:
        base_path = new_cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(cfg_path), base_path)
        return load_config(cfg, base_path)
    return cfg


def main():
    parser = make_parser()
    args = parser.parse_args()

    # TODO: Serialize cfg to a data class (pydantic?)
    cfg = load_config({}, args.config)
    cfg["fit"] = cfg.get("fit", "model" in cfg)
    cfg["sim"] = cfg.get("sim", False)
    cfg["wnoise"] = cfg.get("wnoise", False)
    cfg["map"] = cfg.get("map", True)
    cfg["sub"] = cfg.get("sub", True)
    if args.wnoise:
        cfg["wnoise"] = True
    if args.nosub:
        cfg["sub"] = False
    if args.nofit:
        cfg["fit"] = False

    # Get TODs
    todvec = load_tods(cfg)

    # make a template map with desired pixel size an limits that cover the data
    # todvec.lims() is MPI-aware and will return global limits, not just
    # the ones from private TODs
    lims = todvec.lims()
    pixsize = 2.0 / 3600 * np.pi / 180
    skymap = minkasi.maps.SkyMap(lims, pixsize)

    # Define the model and get stuff setup for minkasi
    model = Model.from_cfg(cfg)
    funs = [model.minkasi_helper]
    params = np.array(model.pars)
    npars = np.array([len(params)])
    prior_vals = model.priors
    priors = [None if prior is None else "flat" for prior in prior_vals]

    # Deal with bowling and simming in TODs and setup noise
    noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
    noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
    noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
    bowl_str = process_tods(
        cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
    )

    # Get output
    outdir = get_outdir(cfg, bowl_str, model)

    # Now we fit
    pars_fit = params.copy()

    print_once("Starting pars: \n")
    for par, label in zip(params, model.par_names):
        print_once(label, ": {:.2e}".format(par))

    if cfg["sim"] and cfg["fit"]:
        params[model.to_fit] *= 1.1  # Don't start at exactly the right value

    if cfg["fit"]:
        for i in range(model.n_rounds):
            model.cur_round = i
            to_fit = np.array(model.to_fit)
            print_once(
                f"Starting round {i+1} of fitting with {np.sum(to_fit)} pars free"
            )
            t1 = time.time()
            (
                pars_fit,
                chisq,
                _,
                errs,
            ) = minkasi.fitting.fit_timestreams_with_derivs_manyfun(
                funs,
                params,
                npars,
                todvec,
                to_fit,
                maxiter=cfg["minkasi"]["maxiter"],
                priors=priors,
                prior_vals=prior_vals,
            )
            minkasi.comm.barrier()
            t2 = time.time()
            print_once("Took", t2 - t1, "seconds to fit")

            model.update(pars_fit, errs, chisq)
            print_once(model)
            params = pars_fit.copy()

            if minkasi.myrank == 0:
                res_path = os.path.join(outdir, f"results_{i}.dill")
                print_once("Saving results to", res_path)
                # TODO: switch to h5?
                model.save(res_path)

            # Reestimate noise
            for i, tod in enumerate(todvec.tods):
                pred = core.model_tod(
                    model.xyz,
                    *model.n_struct,
                    model.dz,
                    model.beam,
                    (tod.info["dx"] - model.x0) * mju.rad_to_arcsec,
                    (tod.info["dy"] - model.y0) * mju.rad_to_arcsec,
                    *params,
                )

                tod.set_noise(
                    noise_class,
                    tod.info["dat_calib"] - pred,
                    *noise_args,
                    **noise_kwargs,
                )
            minkasi.barrier()

    # If we arenn't mapmaking then we can stop here
    if not cfg["map"]:
        return

    # Compute residual and either set it to the data or use it for noise
    for i, tod in enumerate(todvec.tods):
        pred = core.model_tod(
            model.xyz,
            *model.n_struct,
            model.dz,
            model.beam,
            (tod.info["dx"] - model.x0) * mju.rad_to_arcsec,
            (tod.info["dy"] - model.y0) * mju.rad_to_arcsec,
            *params,
        )
        if cfg["sub"]:
            tod.info["dat_calib"] -= np.array(pred)
            tod.set_noise(noise_class, *noise_args, **noise_kwargs)
        else:
            tod.set_noise(
                noise_class, tod.info["dat_calib"] - pred, *noise_args, **noise_kwargs
            )

    # Make maps
    npass = cfg["minkasi"]["npass"]
    dograd = cfg["minkasi"]["dograd"]

    print_once("starting hits and naive")
    naive, hits = mm.make_naive(todvec, skymap, outdir)
    print_once("finished hits and naive")

    # Take 1 over hits map
    ihits = hits.copy()
    ihits.invert()

    # Save weights and noise maps
    _ = mm.make_weights(todvec, skymap, outdir)

    # Setup the mapset
    # For now just include the naive map so we can use it as the initial guess.
    mapset = minkasi.maps.Mapset()
    mapset.add_map(naive)

    # run PCG to solve for a first guess
    iters = [5, 25, 100]
    mapset = mm.solve_map(todvec, mapset, ihits, None, 26, iters, outdir, "initial")

    # Now we iteratively solve and reestimate the noise
    for niter in range(npass):
        maxiter = 26 + 25 * (niter + 1)
        mm.reestimate_noise_from_map(
            todvec, mapset, noise_class, noise_args, noise_kwargs
        )

        # Make a gradient based prior
        if dograd:
            prior, mapset = mm.get_grad_prior(todvec, mapset, hits.copy(), thresh=1.8)
        else:
            prior = None

        # Solve
        mapset = mm.solve_map(
            todvec, mapset, ihits, prior, maxiter, iters, outdir, f"niter_{niter+1}"
        )

    minkasi.barrier()
