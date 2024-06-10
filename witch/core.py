"""
Core module for generating models aed their gradients.
"""

import inspect

import jax
import jax.numpy as jnp

if hasattr(jnp, "trapz"):
    trapz = jnp.trapz
else:
    from jax.scipy.integrate import trapezoid as trapz

import numpy as np

from .structure import (
    N_PAR_A10,
    N_PAR_EGAUSSIAN,
    N_PAR_EXPONENTIAL,
    N_PAR_GAUSSIAN,
    N_PAR_GNFW,
    N_PAR_ISOBETA,
    N_PAR_POWERLAW,
    N_PAR_UNIFORM,
    a10,
    add_exponential,
    add_powerlaw,
    add_powerlaw_cos,
    add_uniform,
    egaussian,
    gaussian,
    gnfw,
    isobeta,
)
from .utils import bilinear_interp, fft_conv

ORDER = (
    "isobeta",
    "gnfw",
    "a10",
    "gaussian",
    "egaussian",
    "uniform",
    "exponential",
    "powerlaw",
    "powerlaw_cos",
)

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")


def _get_static(signature, prefix_list=["n_", "argnums"]):
    par_names = np.array(list(signature.parameters.keys()), dtype=str)
    static_msk = np.zeros_like(par_names, dtype=bool)
    for prefix in prefix_list:
        static_msk += np.char.startswith(par_names, prefix)
    return tuple(np.where(static_msk)[0])


def model(
    xyz,
    n_isobeta,
    n_gnfw,
    n_a10,
    n_gaussian,
    n_egaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dz,
    beam,
    *params,
):
    """
    Generically create models with substructure.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_isobeta: Number of isobeta profiles to add.

        n_gnfw: Number of gnfw profiles to add.

        n_a10: Number of Arnaud2010 profiles to add.

        n_gaussian: Number of gaussians to add.

        n_egaussian: Number of eliptical gaussians to add.

        n_uniform: Number of uniform ellipsoids to add.

        n_exponential: Number of exponential ellipsoids to add.

        n_powerlaw: Number of power law ellipsoids to add.

        n_powerlaw_cos: Number of radial power law ellipsoids with angulas cos term to add.

        dz: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        params: 1D array of model parameters.

    Returns:

        model: The model with the specified substructure evaluated on the grid.
    """
    params = jnp.array(params)
    params = jnp.ravel(params)  # Fixes strange bug with params having dim (1,n)
    isobetas = jnp.zeros((1, 1), dtype=float)
    gnfws = jnp.zeros((1, 1), dtype=float)
    a10s = jnp.zeros((1, 1), dtype=float)
    gaussians = jnp.zeros((1, 1), dtype=float)
    egaussians = jnp.zeros((1, 1), dtype=float)
    uniforms = jnp.zeros((1, 1), dtype=float)
    exponentials = jnp.zeros((1, 1), dtype=float)
    powerlaws = jnp.zeros((1, 1), dtype=float)
    powerlaw_coses = jnp.zeros((1, 1), dtype=float)

    start = 0
    if n_isobeta:
        delta = n_isobeta * N_PAR_ISOBETA
        isobetas = params[start : start + delta].reshape((n_isobeta, N_PAR_ISOBETA))
        start += delta
    if n_gnfw:
        delta = n_gnfw * N_PAR_GNFW
        gnfws = params[start : start + delta].reshape((n_gnfw, N_PAR_GNFW))
        start += delta
    if n_a10:
        delta = n_a10 * N_PAR_A10
        a10s = params[start : start + delta].reshape((n_a10, N_PAR_A10))
        start += delta
    if n_gaussian:
        delta = n_gaussian * N_PAR_GAUSSIAN
        gaussians = params[start : start + delta].reshape((n_gaussian, N_PAR_GAUSSIAN))
        start += delta
    if n_egaussian:
        delta = n_egaussian * N_PAR_EGAUSSIAN
        egaussians = params[start : start + delta].reshape(
            (n_egaussian, N_PAR_EGAUSSIAN)
        )
        start += delta
    if n_uniform:
        delta = n_uniform * N_PAR_UNIFORM
        uniforms = params[start : start + delta].reshape((n_uniform, N_PAR_UNIFORM))
        start += delta
    if n_exponential:
        delta = n_exponential * N_PAR_EXPONENTIAL
        exponentials = params[start : start + delta].reshape(
            (n_exponential, N_PAR_EXPONENTIAL)
        )
        start += delta
    if n_powerlaw:
        delta = n_powerlaw * N_PAR_POWERLAW
        powerlaws = params[start : start + delta].reshape((n_powerlaw, N_PAR_POWERLAW))
        start += delta
    if n_powerlaw_cos:
        delta = n_powerlaw_cos * N_PAR_POWERLAW
        powerlaw_coses = params[start : start + delta].reshape(
            (n_powerlaw_cos, N_PAR_POWERLAW)
        )
        start += delta

    pressure = jnp.zeros((xyz[0].shape[0], xyz[1].shape[1], xyz[2].shape[2]))
    for i in range(n_isobeta):
        pressure = jnp.add(pressure, isobeta(*isobetas[i], xyz))

    for i in range(n_gnfw):
        pressure = jnp.add(pressure, gnfw(*gnfws[i], xyz))

    for i in range(n_a10):
        pressure = jnp.add(pressure, a10(*a10s[i], xyz))

    for i in range(n_gaussian):
        pressure = jnp.add(pressure, gaussian(*gaussians[i], xyz))

    for i in range(n_egaussian):
        pressure = jnp.add(pressure, egaussian(*egaussians[i], xyz))

    for i in range(n_uniform):
        pressure = add_uniform(pressure, xyz, *uniforms[i])

    for i in range(n_exponential):
        pressure = add_exponential(pressure, xyz, *exponentials[i])

    for i in range(n_powerlaw):
        pressure = add_powerlaw(pressure, xyz, *powerlaws[i])

    for i in range(n_powerlaw_cos):
        pressure = add_powerlaw_cos(pressure, xyz, *powerlaw_coses[i])

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)

    bound0, bound1 = int((ip.shape[0] - beam.shape[0]) / 2), int(
        (ip.shape[1] - beam.shape[1]) / 2
    )
    beam = jnp.pad(
        beam,
        (
            (bound0, ip.shape[0] - beam.shape[0] - bound0),
            (bound1, ip.shape[1] - beam.shape[1] - bound1),
        ),
    )

    ip = fft_conv(ip, beam)

    for i in range(n_gaussian):
        ip = jnp.add(ip, gaussian(*gaussians[i], xyz))

    return ip


def model_tod(
    xyz,
    n_isobeta,
    n_gnfw,
    n_a10,
    n_gaussian,
    n_egaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dz,
    beam,
    dx,
    dy,
    *params,
):
    """
    A wrapper around model that unwraps it into a TOD.
    Only the additional arguments are described here, see model for the others.
    Note that the additional arguments are passed **before** the *params argument.

    Arguments:

        dx: RA TOD in units of pixels.
            Should have Dec stretch applied.

        dy: Dec TOD in units of pixels.

    Returns:

        model: The model with the specified substructure.
               Has the same shape as idz.
    """
    ip = model(
        xyz,
        n_isobeta,
        n_gnfw,
        n_a10,
        n_gaussian,
        n_egaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dz,
        beam,
        *params,
    )

    # Assuming xyz is sparse and ij indexed here
    model_out = bilinear_interp(dx, dy, xyz[0].ravel(), xyz[1].ravel(), ip)
    return model_out


def model_grad(
    xyz,
    n_isobeta,
    n_gnfw,
    n_a10,
    n_gaussian,
    n_egaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dz,
    beam,
    argnums,
    *params,
):
    """
    A wrapper around model that also returns the gradients of the model.
    Only the additional arguments are described here, see model for the others.
    Note that the additional arguments are passed **before** the *params argument.

    Arguments:

        argnums: The arguments to evaluate the gradient at

        params: 1D array of model parameters.

    Returns:

        model: The model with the specified substructure.

        grad: The gradient of the model with respect to the model parameters.
    """
    pred = model(
        xyz,
        n_isobeta,
        n_gnfw,
        n_a10,
        n_gaussian,
        n_egaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dz,
        beam,
        *params,
    )

    grad = jax.jacfwd(model, argnums=argnums)(
        xyz,
        n_isobeta,
        n_gnfw,
        n_a10,
        n_gaussian,
        n_egaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dz,
        beam,
        *params,
    )
    grad_padded = jnp.zeros((len(params),) + pred.shape)
    grad_padded = grad_padded.at[jnp.array(argnums) - ARGNUM_SHIFT].set(jnp.array(grad))

    return pred, grad_padded


def model_tod_grad(
    xyz,
    n_isobeta,
    n_gnfw,
    n_a10,
    n_gaussian,
    n_egaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dz,
    beam,
    dx,
    dy,
    argnums,
    *params,
):
    """
    A wrapper around model_tod that also returns the gradients of the model.
    Only the additional arguments are described here, see model for the others.
    Note that the additional arguments are passed **before** the *params argument.

    Arguments:

        dx: RA TOD in units of pixels.
            Should have Dec stretch applied.

        dy: Dec TOD in the same units as xyz.

        argnums: The arguments to evaluate the gradient at

    Returns:

        model: The model with the specified substructure.

        grad: The gradient of the model with respect to the model parameters.
    """
    pred = model_tod(
        xyz,
        n_isobeta,
        n_gnfw,
        n_a10,
        n_gaussian,
        n_egaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dz,
        beam,
        dx,
        dy,
        *params,
    )

    grad = jax.jacfwd(model_tod, argnums=argnums)(
        xyz,
        n_isobeta,
        n_gnfw,
        n_a10,
        n_gaussian,
        n_egaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dz,
        beam,
        dx,
        dy,
        *params,
    )
    grad_padded = jnp.zeros((len(params),) + pred.shape)
    grad_padded = grad_padded.at[jnp.array(argnums) - ARGNUM_SHIFT_TOD].set(
        jnp.array(grad)
    )

    return pred, grad_padded


# Do some signature inspection to avoid hard coding
model_sig = inspect.signature(model)
model_grad_sig = inspect.signature(model_grad)
model_tod_sig = inspect.signature(model_tod)
model_tod_grad_sig = inspect.signature(model_tod_grad)

# Get argnum shifts, -1 is for param
ARGNUM_SHIFT = len(model_sig.parameters) - 1
ARGNUM_SHIFT_TOD = len(model_tod_sig.parameters) - 1

# Figure out static argnums
model_static = _get_static(model_sig)
model_grad_static = _get_static(model_grad_sig)
model_tod_static = _get_static(model_tod_sig)
model_tod_grad_static = _get_static(model_tod_grad_sig)

# Now JIT
model = jax.jit(model, static_argnums=model_static)
model_grad = jax.jit(model_grad, static_argnums=model_grad_static)
model_tod = jax.jit(model_tod, static_argnums=model_tod_static)
model_tod_grad = jax.jit(model_tod_grad, static_argnums=model_tod_grad_static)