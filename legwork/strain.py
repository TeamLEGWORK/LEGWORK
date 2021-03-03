"""Computes several types of gravitational wave strains"""

import astropy.constants as c
from legwork import utils
import numpy as np

__all__ = ['h_0_n', 'h_c_n']


def h_0_n(m_c, f_orb, ecc, n, dist, interpolated_g=None):
    """Computes strain amplitude

    Computes the dimensionless power of a general binary
    radiating gravitational waves in the quadrupole approximation
    at ``n`` harmonics of the orbital frequency

    In the docs below, `x` refers to the number of sources, `y` to the number
    of timesteps and `z` to the number of harmonics.

    Parameters
    ----------
    m_c : `float/array`
        Chirp mass of each binary. Shape should be (x,).

    f_orb : `float/array`
        Orbital frequency of each binary at each timestep.
        Shape should be (x, y), or (x,) if only one timestep.

    ecc : `float/array`
        Eccentricity of each binary at each timestep. Shape should be (x, y),
        or (x,) if only one timestep.

    n : `int/array`
        Harmonic(s) at which to calculate the strain. Either a single int or
        shape should be (z,)

    dist : `float/array`
        Distance to each binary. Shape should be (x,)

    interpolated_g : `function`
        A function returned by :class:`scipy.interpolate.interp2d` that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    h_0 : `float/array`
        Strain amplitude. Shape is (x, y, z).
    """
    # convert to array if necessary
    arrayed_args, _ = utils.ensure_array(m_c, f_orb, ecc, n, dist)
    m_c, f_orb, ecc, n, dist = arrayed_args

    # if one timestep then extend dimensions
    if f_orb.ndim != 2:
        f_orb = f_orb[:, np.newaxis]
    if ecc.ndim != 2:
        ecc = ecc[:, np.newaxis]

    # extend mass and distance dimensions
    m_c = m_c[:, np.newaxis]
    dist = dist[:, np.newaxis]

    # work out strain for n independent part and broadcast to correct shape
    prefac = (2**(28/3) / 5)**(0.5) * c.G**(5/3) / c.c**4
    n_independent_part = prefac * m_c**(5/3) * (np.pi * f_orb)**(2/3) / dist

    # check whether to interpolate g(n, e)
    if interpolated_g is None:
        # extend harmonic and eccentricity dimensions to full (x, y, z)
        n = n[np.newaxis, np.newaxis, :]
        ecc = ecc[..., np.newaxis]
        n_dependent_part = utils.peters_g(n, ecc)**(1/2) / n
    else:
        # flatten array to work nicely interp2d
        g_vals = interpolated_g(n, ecc.flatten())

        # set negative values from cubic fit to 0.0
        g_vals[g_vals < 0.0] = 0.0

        # unsort the output array if there is more than one eccentricity
        if isinstance(ecc, (np.ndarray, list)) and len(ecc) > 1:
            g_vals = g_vals[np.argsort(ecc.flatten()).argsort()]

        # reshape output to proper dimensions
        g_vals = g_vals.reshape((*ecc.shape, len(n)))

        n_dependent_part = g_vals**(0.5) / n[np.newaxis, np.newaxis, :]

    h_0 = n_independent_part[..., np.newaxis] * n_dependent_part

    return h_0.decompose()


def h_c_n(m_c, f_orb, ecc, n, dist, interpolated_g=None):
    """Computes characteristic strain amplitude

    Computes the dimensionless characteristic power of a general
    binary radiating gravitational waves in the quadrupole approximation
    at the nth harmonic of the orbital frequency.

    In the docs below, `x` refers to the number of sources, `y` to the number
    of timesteps and `z` to the number of harmonics.

    Parameters
    ----------
    m_c : `float/array`
        Chirp mass of each binary. Shape should be (x,).

    f_orb : `float/array`
        Orbital frequency of each binary at each timestep.
        Shape should be (x, y), or (x,) if only one timestep.

    ecc : `float/array`
        Eccentricity of each binary at each timestep. Shape should be (x, y),
        or (x,) if only one timestep.

    n : `int/array`
        Harmonic(s) at which to calculate the strain. Either a single int or
        shape should be (z,)

    dist : `float/array`
        Distance to each binary. Shape should be (x,)

    interpolated_g : `function`
        A function returned by :class:`scipy.interpolate.interp2d` that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    h_c : `float/array`
        Characteristic strain. Shape is (x, y, z).
    """
    # convert to array if necessary
    arrayed_args, _ = utils.ensure_array(m_c, f_orb, ecc, n, dist)
    m_c, f_orb, ecc, n, dist = arrayed_args

    # if one timestep then extend dimensions
    if f_orb.ndim != 2:
        f_orb = f_orb[:, np.newaxis]
    if ecc.ndim != 2:
        ecc = ecc[:, np.newaxis]

    # extend mass and distance dimensions
    m_c = m_c[:, np.newaxis]
    dist = dist[:, np.newaxis]

    # work out strain for n independent part
    prefac = (2**(5/3) / (3 * np.pi**(4/3)))**(0.5) * c.G**(5/6) / c.c**(3/2)
    n_independent_part = prefac * m_c**(5/6) / dist * f_orb**(-1/6) \
        / utils.peters_f(ecc)**(0.5)

    # check whether to interpolate g(n, e)
    if interpolated_g is None:
        # extend harmonic and eccentricity dimensions to full (x, y, z)
        n = n[np.newaxis, np.newaxis, :]
        ecc = ecc[..., np.newaxis]
        n_dependent_part = (utils.peters_g(n, ecc) / n)**(1/2)
    else:
        # flatten array to work nicely interp2d
        g_vals = interpolated_g(n, ecc.flatten())

        # set negative values from cubic fit to 0.0
        g_vals[g_vals < 0.0] = 0.0

        # unsort the output array if there is more than one eccentricity
        if isinstance(ecc, (np.ndarray, list)) and len(ecc) > 1:
            g_vals = g_vals[np.argsort(ecc.flatten()).argsort()]

        # reshape output to proper dimensions
        g_vals = g_vals.reshape((*ecc.shape, len(n)))

        n_dependent_part = (g_vals / n[np.newaxis, np.newaxis, :])**(0.5)

    h_c = n_independent_part[..., np.newaxis] * n_dependent_part
    return h_c.decompose()
