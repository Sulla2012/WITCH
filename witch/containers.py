"""
Data classes for describing models in a structured way
"""

from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module
from typing import Optional

import dill
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from minkasi.tods import Tod
from numpy.typing import NDArray
from typing_extensions import Self

from . import core
from . import utils as wu
from .structure import STRUCT_N_PAR


@dataclass
class Parameter:
    name: str
    fit: list[bool]
    val: float
    err: float = 0
    prior: Optional[scipy.stats.rv_continuous] = None

    @property
    def fit_ever(self) -> bool:
        return bool(np.any(self.fit))


@dataclass
class Structure:
    name: str
    structure: str
    parameters: list[Parameter]

    def __post_init__(self):
        self.structure = self.structure.lower()
        # Check that this is a valid structure
        if self.structure not in STRUCT_N_PAR.keys():
            raise ValueError(f"{self.name} has invalid structure: {self.structure}")
        # Check that we have the correct number of params
        if len(self.parameters) != STRUCT_N_PAR[self.structure]:
            raise ValueError(
                f"{self.name} has incorrect number of parameters, expected {STRUCT_N_PAR[self.structure]} for {self.structure} but was given {len(self.parameters)}"
            )


@dataclass
class Model:
    name: str
    structures: list[Structure]
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float]  # arcseconds
    dz: float  # arcseconds * unknown
    beam: jax.Array
    n_rounds: int
    cur_round: int = 0
    chisq: float = np.inf
    original_order: list[int] = field(init=False)

    def __post_init__(self):
        # Make sure the structure is in the order that core expects
        structure_idx = np.argsort(
            [core.ORDER.index(structure.structure) for structure in self.structures]
        )
        self.structures = [self.structures[i] for i in structure_idx]
        self.original_order = list(np.sort(structure_idx))

    def __set_attr__(self, name, value):
        if name == "cur_round":
            self.__dict__.pop("model_grad", None)
        return super().__setattr__(name, value)

    @cached_property
    def n_struct(self) -> list[int]:
        n_struct = [0] * len(core.ORDER)
        for structure in self.structures:
            idx = core.ORDER.index(structure.structure)
            n_struct[idx] += 1
        return n_struct

    @property
    def pars(self) -> list[float]:
        pars = []
        for structure in self.structures:
            pars += [parameter.val for parameter in structure.parameters]
        return pars

    @cached_property
    def par_names(self) -> list[str]:
        par_names = []
        for structure in self.structures:
            par_names += [parameter.name for parameter in structure.parameters]
        return par_names

    @property
    def errs(self) -> list[float]:
        errs = []
        for structure in self.structures:
            errs += [parameter.err for parameter in structure.parameters]
        return errs

    @cached_property
    def priors(self) -> list[Optional[tuple[float, float]]]:
        priors = []
        for structure in self.structures:
            priors += [parameter.prior for parameter in structure.parameters]
        return priors

    @property
    def to_fit(self) -> list[bool]:
        to_fit = []
        for structure in self.structures:
            to_fit += [
                parameter.fit[self.cur_round] for parameter in structure.parameters
            ]
        return to_fit

    @cached_property
    def to_fit_ever(self) -> list[bool]:
        to_fit = []
        for structure in self.structures:
            to_fit += [parameter.fit_ever for parameter in structure.parameters]
        return to_fit

    @cached_property
    def model(self) -> jax.Array:
        return core.model(
            self.xyz,
            tuple(self.n_struct),
            self.dz,
            self.beam,
            *self.pars,
        )

    def to_tod(self, dx, dy) -> jax.Array:
        """
        Project the model into a TOD.

        Arguments:

            dx: The RA TOD in arcseconds.

            dy: The Dec TOD in arcseconds.

        Returns:

            tod: The model as a TOD.
                 Same shape as dx.
        """
        return wu.bilinear_interp(
            dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), self.model
        )

    @cached_property
    def model_grad(self) -> tuple[jax.Array, jax.Array]:
        argnums = tuple(np.where(self.to_fit)[0] + core.ARGNUM_SHIFT)
        return core.model_grad(
            self.xyz,
            tuple(self.n_struct),
            self.dz,
            self.beam,
            argnums,
            *self.pars,
        )

    def to_tod_grad(self, dx, dy) -> tuple[jax.Array, jax.Array]:
        """
        Project the model and gradient into a TOD.

        Arguments:

            dx: The RA TOD in arcseconds.

            dy: The Dec TOD in arcseconds.

        Returns:

            tod: The model as a TOD.
                 Same shape as dx.

            grad_tod: The gradient a TOD.
                      Has shape (npar,) + dx.shape
        """
        model, grad = self.model_grad
        tod = wu.bilinear_interp(
            dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), model
        )
        grad_tod = jnp.array(
            [
                (
                    wu.bilinear_interp(
                        dx, dy, self.xyz[0].ravel(), self.xyz[1].ravel(), _grad
                    )
                    if _fit
                    else jnp.zeros_like(tod)
                )
                for _grad, _fit in zip(grad, self.to_fit)
            ]
        )

        return tod, grad_tod

    def __repr__(self) -> str:
        rep = self.name + ":\n"
        rep += f"Round {self.cur_round + 1} out of {self.n_rounds}\n"
        for i in self.original_order:
            struct = self.structures[i]
            rep += "\t" + struct.name + ":\n"
            for par in struct.parameters:
                rep += (
                    "\t\t"
                    + par.name
                    + "*" * par.fit[self.cur_round]
                    + str(par.prior) * (par.prior is not None)
                    + " = "
                    + str(par.val)
                    + " ± "
                    + str(par.err)
                    + "\n"
                )
        rep += f"chisq is {self.chisq}"
        return rep

    def update(self, vals, errs, chisq):
        if not np.array_equal(self.pars, vals):
            self.__dict__.pop("model", None)
            self.__dict__.pop("model_grad", None)
        n = 0
        for struct in self.structures:
            for par in struct.parameters:
                par.val = vals[n]
                par.err = errs[n]
                n += 1
        self.chisq = chisq

    def minkasi_helper(
        self, params: NDArray[np.floating], tod: Tod
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Helper function to work with minkasi fitting routines.

        Arguments:

            params: An array of model parameters.

            tod: A minkasi tod instance.
                'dx' and 'dy' must be in tod.info and be in radians.

        Returns:

            grad: The gradient of the model with respect to the model parameters.

            pred: The model with the specified substructure.
        """
        self.update(params, self.errs, self.chisq)
        dx = tod.info["dx"] * wu.rad_to_arcsec
        dy = tod.info["dy"] * wu.rad_to_arcsec

        pred_tod, grad_tod = self.to_tod_grad(dx, dy)
        pred_tod = jax.device_get(pred_tod)
        grad_tod = jax.device_get(grad_tod)

        return grad_tod, pred_tod

    def save(self, path: str):
        """
        Serialize the model to a file with dill.

        Arguments:

            path: The file to save to.
        """
        with open(path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load the model from a file with dill.

        Arguments:

            path: The path to the saved model
        """
        with open(path, "rb") as f:
            return dill.load(f)

    @classmethod
    def from_cfg(cls, cfg: dict) -> Self:
        """
        Create an instance of model from a witch config.

        Arguments:

            cfg: Config loaded into a dict.
        """
        # Do imports
        for module, name in cfg.get("imports", {}).items():
            mod = import_module(module)
            if isinstance(name, str):
                locals()[name] = mod
            elif isinstance(name, list):
                for n in name:
                    locals()[n] = getattr(mod, n)
            else:
                raise TypeError("Expect import name to be a string or a list")

        # Load constants
        constants = {
            name: eval(str(const)) for name, const in cfg.get("constants", {}).items()
        }  # pyright: ignore [reportUnusedVariable]

        # Get jax device
        dev_id = cfg.get("jax_device", 0)
        device = jax.devices()[dev_id]

        # Setup coordindate stuff
        r_map = eval(str(cfg["coords"]["r_map"]))
        dr = eval(str(cfg["coords"]["dr"]))
        dz = eval(str(cfg["coords"].get("dz", dr)))
        x0 = eval(str(cfg["coords"]["x0"]))
        y0 = eval(str(cfg["coords"]["y0"]))

        xyz_host = wu.make_grid(
            r_map, dr, dr, dz, x0 * wu.rad_to_arcsec, y0 * wu.rad_to_arcsec
        )
        xyz = jax.device_put(xyz_host, device)
        xyz[0].block_until_ready()
        xyz[1].block_until_ready()
        xyz[2].block_until_ready()

        # Make beam
        beam = wu.beam_double_gauss(
            dr,
            eval(str(cfg["beam"]["fwhm1"])),
            eval(str(cfg["beam"]["amp1"])),
            eval(str(cfg["beam"]["fwhm2"])),
            eval(str(cfg["beam"]["amp2"])),
        )
        beam = jax.device_put(beam, device)

        n_rounds = cfg.get("n_rounds", 1)
        dz = dz * eval(str(cfg["model"]["unit_conversion"]))

        structures = []
        for name, structure in cfg["model"]["structures"].items():
            parameters = []
            for par_name, param in structure["parameters"].items():
                val = eval(str(param["value"]))
                fit = param.get("to_fit", [False] * n_rounds)
                if isinstance(fit, bool):
                    fit = [fit] * n_rounds
                if len(fit) != n_rounds:
                    raise ValueError(
                        f"to_fit has {len(fit)} entries but we only have {n_rounds} rounds"
                    )
                priors = param.get("priors", None)
                if priors is not None:
                    if isinstance(priors, list):
                        priors = scipy.stats.uniform(
                            loc=priors[0], scale=priors[1] - priors[0]
                        )
                    else:
                        priors = eval("scipy.stats." + str(priors))
                parameters.append(Parameter(par_name, fit, val, 0.0, priors))
            structures.append(Structure(name, structure["structure"], parameters))
        name = cfg["model"].get(
            "name", "-".join([structure.name for structure in structures])
        )

        return cls(name, structures, xyz, dz, beam, n_rounds)
