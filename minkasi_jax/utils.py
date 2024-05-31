"""
A set of utility functions and constants used for unit conversions
and adding generic structure common to multiple models.
"""

from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from jax.typing import ArrayLike
from minkasi.maps import SkyMap
from numpy.typing import NDArray

AnyArray = jax.Array | NDArray

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")
# Constants
# --------------------------------------------------------

ap = 0.12
h70 = cosmo.H0.value / 7.00e01

Tcmb = 2.7255
kb = const.k_B.value
me = ((const.m_e * const.c**2).to(u.keV)).value
h = const.h.value
Xthom = const.sigma_T.to(u.cm**2).value

Mparsec = u.Mpc.to(u.cm)
XMpc = Xthom * Mparsec

rad_to_arcsec = (180 * 3600) / np.pi

# Cosmology
# --------------------------------------------------------
dzline = np.linspace(0.00, 5.00, 1000)
daline = cosmo.angular_diameter_distance(dzline) / u.radian
nzline = cosmo.critical_density(dzline)
hzline = cosmo.H(dzline) / cosmo.H0

daline = daline.to(u.Mpc / u.arcsec)
nzline = nzline.to(u.Msun / u.Mpc**3)

dzline = jnp.array(dzline)
hzline = jnp.array(hzline.value)
nzline = jnp.array(nzline.value)
daline = jnp.array(daline.value)


# Unit conversions
# --------------------------------------------------------
@partial(jax.jit, static_argnums=(0, 1))
def y2K_CMB(freq: float, Te: float) -> float:
    """Convert from compton y to K_CMB.

    Parameters
    ----------
    freq :
        The observing frequency in Hz
    Te :
        Electron temperature
    freq: float :

    Te: float :


    Returns
    -------
    type
        y2K_CMB: Conversion factor from compton y to K_CMB.

    """
    x = freq * h / kb / Tcmb
    xt = x / jnp.tanh(0.5 * x)
    st = x / jnp.sinh(0.5 * x)
    # fmt:off
    Y0 = -4.0 + xt
    Y1 = (-10.0
        + ((47.0 / 2.0) + (-(42.0 / 5.0) + (7.0 / 10.0) * xt) * xt) * xt
        + st * st * (-(21.0 / 5.0) + (7.0 / 5.0) * xt)
    )
    Y2 = ((-15.0 / 2.0)
        + ((1023.0 / 8.0) + ((-868.0 / 5.0) + ((329.0 / 5.0) + ((-44.0 / 5.0) + (11.0 / 30.0) * xt) * xt) * xt) * xt) * xt
        + ((-434.0 / 5.0) + ((658.0 / 5.0) + ((-242.0 / 5.0) + (143.0 / 30.0) * xt) * xt) * xt
        + (-(44.0 / 5.0) + (187.0 / 60.0) * xt) * (st * st)) * st * st
    )
    Y3 = ((15.0 / 2.0)
        + ((2505.0 / 8.0) + ((-7098.0 / 5.0) + ((14253.0 / 10.0) + ((-18594.0 / 35.0) 
         + ((12059.0 / 140.0) + ((-128.0 / 21.0) + (16.0 / 105.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt
        + (((-7098.0 / 10.0) + ((14253.0 / 5.0) + ((-102267.0 / 35.0) + ((156767.0 / 140.0)
         + ((-1216.0 / 7.0) + (64.0 / 7.0) * xt) * xt) * xt) * xt) * xt)
         + (((-18594.0 / 35.0) + ((205003.0 / 280.0) + ((-1920.0 / 7.0) + (1024.0 / 35.0) * xt) * xt) * xt)
          + ((-544.0 / 21.0) + (992.0 / 105.0) * xt) * st * st) * st * st) * st * st
    )
    Y4 = ((-135.0 / 32.0)
        + ((30375.0 / 128.0) + ((-62391.0 / 10.0) + ((614727.0 / 40.0) + ((-124389.0 / 10.0) + ((355703.0 / 80.0) + ((-16568.0 / 21.0)
         + ((7516.0 / 105.0) + ((-22.0 / 7.0) + (11.0 / 210.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt) * xt) * xt
        + ((-62391.0 / 20.0) + ((614727.0 / 20.0) + ((-1368279.0 / 20.0) + ((4624139.0 / 80.0) + ((-157396.0 / 7.0) + ((30064.0 / 7.0)
         + ((-2717.0 / 7.0) + (2761.0 / 210.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt
         + ((-124389.0 / 10.0)
          + ((6046951.0 / 160.0) + ((-248520.0 / 7.0) + ((481024.0 / 35.0) + ((-15972.0 / 7.0) + (18689.0 / 140.0) * xt) * xt) * xt) * xt) * xt
          + ((-70414.0 / 21.0) + ((465992.0 / 105.0) + ((-11792.0 / 7.0) + (19778.0 / 105.0) * xt) * xt) * xt
           + ((-682.0 / 7.0) + (7601.0 / 210.0) * xt) * st * st) * st * st) * st * st) * st * st
    )
    # fmt:on
    factor = Y0 + (Te / me) * (
        Y1 + (Te / me) * (Y2 + (Te / me) * (Y3 + (Te / me) * Y4))
    )
    return factor * Tcmb


@partial(jax.jit, static_argnums=(0,))
def K_CMB2K_RJ(freq: float) -> float:
    """Convert from K_CMB to K_RJ.

    Parameters
    ----------
    freq :
        The observing frequency in Hz
    freq: float :


    Returns
    -------
    type
        K_CMB2K_RJ: Conversion factor from K_CMB to K_RJ.

    """
    x = freq * h / kb / Tcmb
    return jnp.exp(x) * x * x / jnp.expm1(x) ** 2


@partial(jax.jit, static_argnums=(0, 1))
def y2K_RJ(freq: float, Te: float) -> float:
    """Convert from compton y to K_RJ.

    Parameters
    ----------
    freq :
        The observing frequency in Hz
    Te :
        Electron temperature
    freq: float :

    Te: float :


    Returns
    -------
    type
        y2K_RJ: Conversion factor from compton y to K_RJ.

    """
    factor = y2K_CMB(freq, Te)
    return factor * K_CMB2K_RJ(freq)


def get_da(z: float) -> float:
    """Get factor to convert from arcseconds to MPc.

    Parameters
    ----------
    z :
        The redshift at which to compute the factor
    z: float :


    Returns
    -------
    type
        da: Conversion factor from arcseconds to MPc

    """
    return jnp.interp(z, dzline, daline).item()


def get_nz(z: float) -> float:
    """Get n(z).

    Parameters
    ----------
    z :
        The redshift at which to compute the factor
    z: float :


    Returns
    -------
    type
        nz: n at the given z.

    """
    return jnp.interp(z, dzline, nzline).item()


def get_hz(z: float) -> float:
    """Get h(z).

    Parameters
    ----------
    z :
        The redshift at which to compute the factor
    z: float :


    Returns
    -------
    type
        hz: h at the given z.

    """
    return jnp.interp(z, dzline, hzline).item()


# FFT Operations
# -----------------------------------------------------------
@jax.jit
def fft_conv(image: ArrayLike, kernel: ArrayLike) -> jax.Array:
    """Perform a convolution using FFTs for speed.

    Parameters
    ----------
    image :
        Data to be convolved
    kernel :
        Convolution kernel
    image: ArrayLike :

    kernel: ArrayLike :


    Returns
    -------
    type
        convolved_map: Image convolved with kernel.

    """
    Fmap = jnp.fft.fft2(jnp.fft.fftshift(image))
    Fkernel = jnp.fft.fft2(jnp.fft.fftshift(kernel))
    convolved_map = jnp.fft.fftshift(jnp.real(jnp.fft.ifft2(Fmap * Fkernel)))

    return convolved_map


@partial(jax.jit, static_argnums=(1,))
def tod_hi_pass(tod: ArrayLike, N_filt: int) -> jax.Array:
    """High pass a tod with a tophat

    Parameters
    ----------
    tod :
        TOD to high pass
    N_filt :
        N_filt of tophat
    tod: ArrayLike :

    N_filt: int :


    Returns
    -------
    type
        tod_filtered: Filtered TOD

    """
    mask = jnp.ones(tod.shape)
    mask = jax.ops.index_update(mask, jax.ops.index[..., :N_filt], 0.0)

    ## apply the filter in fourier space
    Ftod = jnp.fft.fft(tod)
    Ftod_filtered = Ftod * mask
    tod_filtered = jnp.fft.ifft(Ftod_filtered).real
    return tod_filtered


# Model building tools
# -----------------------------------------------------------
def make_grid(
    r_map: float, dx: float, dy: Optional[float] = None, dz: Optional[float] = None
) -> list[jax.Array]:
    """Make coordinate grids to build models in.
    All grids are sparse and are int(2*r_map / dr) in each dimension.

    Parameters
    ----------
    r_map :
        Size of grid radially
    dx :
        Grid resolution in x
    dy :
        Grid resolution in y
    If :
        None then dy is set to dx
    dz :
        Grid resolution in z
    If :
        None then dz is set to dx
    r_map: float :

    dx: float :

    dy: Optional[float] :
         (Default value = None)
    dz: Optional[float] :
         (Default value = None)

    Returns
    -------
    type
        x: Grid of x coordinates in same units as r_map.

        y: Grid of y coordinates in same units as r_map

        z: Grid of z coordinates in same units as r_map

    """
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    # Make grid with resolution dr and size r_map
    x = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dx))
    y = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dy))
    z = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dz))

    return jnp.meshgrid(x, y, z, sparse=True, indexing="ij")


def make_grid_from_skymap(
    skymap: SkyMap,
    z_map: float,
    dz: float,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
):
    """Make coordinate grids to build models in that matches a map's footprint.
    All grids are sparse and are int(2*r_map / dr) in each dimension.

    Parameters
    ----------
    skymap :
        The map to match the grid to
    z_map :
        Size of grid along LOS
    dz :
        Grid resolution along LOS
    x0 :
        Map x center in radians
    y0 :
        Map y center in radians
    skymap: SkyMap :

    z_map: float :

    dz: float :

    x0: Optional[float] :
         (Default value = None)
    y0: Optional[float] :
         (Default value = None)

    Returns
    -------
    type
        x: Grid of x coordinates in radians.

        y: Grid of y coordinates in radians.

        z: Grid of z coordinates in radians.

    """
    # make grid
    _x = jnp.arange(skymap.nx, dtype=float)
    _y = jnp.arange(skymap.ny, dtype=float)
    _z = jnp.linspace(-1 * z_map, z_map, 2 * int(z_map / dz), dtype=float)
    x, y, z = jnp.meshgrid(_x, _y, _z, sparse=True, indexing="ij")

    # Pad so we don't need to broadcast
    x_flat = x.ravel()
    y_flat = y.ravel()
    len_diff = len(x_flat) - len(y_flat)
    if len_diff > 0:
        y_flat = jnp.pad(y_flat, (0, len_diff), "edge")
    elif len_diff < 0:
        x_flat = jnp.pad(x_flat, (0, abs(len_diff)), "edge")

    # Convert x and y to ra/dec
    ra_dec = skymap.wcs.wcs_pix2world(
        jnp.column_stack((x_flat, y_flat)), 0, ra_dec_order=True
    )
    ra_dec = np.deg2rad(ra_dec)
    ra = ra_dec[:, 0]
    dec = ra_dec[:, 1]

    # Remove padding
    if len_diff > 0:
        dec = dec[: (-1 * len_diff)]
    elif len_diff < 0:
        ra = ra[:len_diff]

    if not x0:
        x0 = (skymap.lims[1] + skymap.lims[0]) / 2
    if not y0:
        y0 = (skymap.lims[3] + skymap.lims[2]) / 2

    ra -= x0
    dec -= y0

    # Sparse indexing to save mem
    x = x.at[:, 0, 0].set(ra)
    y = y.at[0, :, 0].set(dec)

    return x, y, z


@jax.jit
def transform_grid(
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    xyz: Sequence[ArrayLike],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Shift, rotate, and apply ellipticity to coordinate grid.

    Parameters
    ----------
    dx :
        RA of cluster center relative to grid origin
    dy :
        Dec of cluster center relative to grid origin
    dz :
        Line of sight offset of cluster center relative to grid origin
    r_1 :
        Amount to scale along x
    r_2 :
        Amount to scale along y
    r_3 :
        Amount to scale along z
    theta :
        Angle to rotate in xy
    xyz :
        Coordinte grid to transform
    dx: float :

    dy: float :

    dz: float :

    r_1: float :

    r_2: float :

    r_3: float :

    theta: float :

    xyz: Sequence[ArrayLike] :


    Returns
    -------
    type
        xyz: Transformed coordinate grid

    """
    # Shift origin
    x = xyz[0] - dx
    y = xyz[1] - dy
    z = xyz[2] - dz

    # Rotate
    xx = x * jnp.cos(theta) + y * jnp.sin(theta)
    yy = y * jnp.cos(theta) - x * jnp.sin(theta)

    # Apply ellipticity
    x = xx / r_1
    y = yy / r_2
    z = z / r_3

    return jnp.array(x), jnp.array(y), jnp.array(z)


@jax.jit
def tod_to_index(
    xi: AnyArray,
    yi: AnyArray,
    x0: float,
    y0: float,
    grid: Sequence[AnyArray],
    conv_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Convert RA/Dec TODs to index space.

    Parameters
    ----------
    xi :
        RA TOD
    yi :
        Dec TOD
    x0 :
        RA at center of model
    y0 :
        Dec at center of model
    grid :
        The grid to index on
    conv_factor :
        Conversion factor to put RA and Dec in same units as r_map
    Nominally :
        da
    xi: AnyArray :

    yi: AnyArray :

    x0: float :

    y0: float :

    grid: Sequence[AnyArray] :

    conv_factor: float :


    Returns
    -------
    type
        idx: The RA TOD in index space

        idy: The Dec TOD in index space.

    """
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= conv_factor
    dy *= conv_factor

    # Assuming sparse indexing here
    idx = jnp.digitize(dx, grid[0].ravel())
    idy = jnp.digitize(dy, grid[1].ravel())

    idx = jnp.rint(idx).astype(int)
    idy = jnp.rint(idy).astype(int)

    # Ensure out of bounds for stuff not in grid
    idx = jnp.where((idx < 0) + (idx >= grid[0].shape[0]), 2 * grid[0].shape[0], idx)
    idy = jnp.where((idy < 0) + (idy >= grid[1].shape[1]), 2 * grid[1].shape[1], idy)

    return idx, idy


@jax.jit
def bilinear_interp(
    x: AnyArray, y: AnyArray, xp: AnyArray, yp: AnyArray, fp: AnyArray
) -> jax.Array:
    """JAX implementation of bilinear interpolation.
    Out of bounds values are set to 0.
    Using the repeated linear interpolation method here,
    see https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation.

    Parameters
    ----------
    x :
        X values to return interpolated values at
    y :
        Y values to return interpolated values at
    xp :
        X values to interpolate with
    Assumed :
        to be sorted
    yp :
        Y values to interpolate with
    Assumed :
        to be sorted
    fp :
        Functon values at
    Note :
        that if you are using meshgrid
    x: AnyArray :

    y: AnyArray :

    xp: AnyArray :

    yp: AnyArray :

    fp: AnyArray :


    Returns
    -------
    type
        f: The interpolated values

    """
    if len(xp.shape) != 1:
        raise ValueError("xp must be 1D")
    if len(yp.shape) != 1:
        raise ValueError("yp must be 1D")
    if fp.shape != xp.shape + yp.shape:
        raise ValueError(
            "Incompatible shapes for fp, xp, yp: %s, %s, %s",
            fp.shape,
            xp.shape,
            yp.shape,
        )

    # Figure out bounds and mapping
    # This breaks if xp, yp is not sorted
    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)
    q_11 = fp[ix - 1, iy - 1]
    q_21 = fp[ix, iy - 1]
    q_12 = fp[ix - 1, iy]
    q_22 = fp[ix, iy]

    # Interpolate in x to start
    denom_x = xp[ix] - xp[ix - 1]
    dx_1 = x - xp[ix - 1]
    dx_2 = xp[ix] - x
    f_xy1 = (dx_2 * q_11 + dx_1 * q_21) / denom_x
    f_xy2 = (dx_2 * q_12 + dx_1 * q_22) / denom_x

    # Now do y as well
    denom_y = yp[iy] - yp[iy - 1]
    dy_1 = y - yp[iy - 1]
    dy_2 = yp[iy] - y
    f = (dy_2 * f_xy1 + dy_1 * f_xy2) / denom_y

    # Zero out the out of bounds values
    f = jnp.where((x < xp[0]) + (x > xp[-1]) + (y < yp[0]) + (y > yp[-1]), 0.0, f)

    return f


def beam_double_gauss(
    dr: float,
    fwhm1: float = 9.735,
    amp1: float = 0.9808,
    fwhm2: float = 32.627,
    amp2: float = 0.0192,
):
    """Helper function to generate a double gaussian beam.

    Parameters
    ----------
    dr :
        Pixel size
    fwhm1 :
        Full width half max of the primary gaussian
    amp1 :
        Amplitude of the primary gaussian
    fwhm2 :
        Full width half max of the secondairy gaussian
    amp2 :
        Amplitude of the secondairy gaussian
    dr: float :

    fwhm1: float :
         (Default value = 9.735)
    amp1: float :
         (Default value = 0.9808)
    fwhm2: float :
         (Default value = 32.627)
    amp2: float :
         (Default value = 0.0192)

    Returns
    -------
    type
        beam: Double gaussian beam.

    """
    x = jnp.arange(-1.5 * fwhm1 // (dr), 1.5 * fwhm1 // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x, x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1 * jnp.exp(-4 * jnp.log(2) * beam_rr**2 / fwhm1**2) + amp2 * jnp.exp(
        -4 * jnp.log(2) * beam_rr**2 / fwhm2**2
    )
    return beam / jnp.sum(beam)
