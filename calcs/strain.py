"""Computes several types of gravitational wave strains"""

import astropy.constants as c
from calcs.utils import peters_g, peters_f
import numpy as np


def h_0_n(m_c, f_orb, ecc, n, dist, interpolated_g=None):
    """Computes the dimensionless power of a general binary
    radiating gravitational waves in the quadrupole approximation
    at the nth harmonic of the orbital frequency

    Params
    ------
    m_c : `float/array`
        chirp mass of the binary

    f_orb : `float/array`
        orbital frequency

    ecc : `float/array`
        eccentricity

    n : `int/array`
        harmonic(s) at which to calculate the strain

    dist : `float/array`
        distance to the binary

    interpolated_g : `function`
        A function returned by scipy.interpolate.interp2d that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    h_0 : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
        shape of array is `(number of sources, number of harmonics)`
    """
    # calculate how many harmonics and sources
    n_harmonics = 1 if isinstance(n, (int, np.int64, np.int)) else len(n)
    n_sources = len(f_orb) if isinstance(f_orb.value, (list, np.ndarray))else 1

    # work out strain for n independent part and broadcast to correct shape
    prefac = (2**(28/3) / 5)**(0.5) * c.G**(5/3) / c.c**4
    n_independent_part = prefac * m_c**(5/3) * (np.pi * f_orb)**(2/3) / dist

    # broadcast to correct shape if necessary
    if n_independent_part.shape != (n_sources, n_harmonics):
        n_independent_part = np.broadcast_to(n_independent_part.decompose(),
                                             (n_harmonics, n_sources)).T

    N, E = np.meshgrid(n, ecc)

    if interpolated_g is None:
        n_dependent_part = peters_g(N, E)**(1/2) / N
    else:
        g_vals = interpolated_g(n, ecc)

        # unsort the output array if there is more than one eccentricity
        if isinstance(ecc, (np.ndarray, list)) and len(ecc) > 1:
            g_vals = g_vals[np.argsort(ecc).argsort()]
        n_dependent_part = g_vals**(0.5) / N

    h_0 = n_independent_part * n_dependent_part
    return h_0


def h_c_n(m_c, f_orb, ecc, n, dist, interpolated_g=None):
    """Computes the dimensionless characteristic power of a general
    binary radiating gravitational waves in the quadrupole approximation
    at the nth harmonic of the orbital frequency

    Params
    ------
    m_c : `float/array`
        chirp mass of the binary

    f_orb : `float/array`
        orbital frequency

    ecc : `float/array`
        eccentricity

    n : `int/array`
        harmonic(s) at which to calculate the strain

    dist : `float/array`
        distance to the binary

    t_obs : `float/array`
        observation duration

    interpolated_g : `function`
        A function returned by scipy.interpolate.interp2d that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    h_c : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """
    # calculate how many harmonics and sources
    n_harmonics = 1 if isinstance(n, (int, np.int64, np.int)) else len(n)
    n_sources = len(f_orb) if isinstance(f_orb.value, (list, np.ndarray))else 1

    # work out strain for n independent part
    prefac = (2**(5/3) / (3 * np.pi**(4/3)))**(0.5) * c.G**(5/6) / c.c**(3/2)
    n_independent_part = prefac * m_c**(5/6) / dist * f_orb**(-1/6) \
                                / peters_f(ecc)**(0.5)

    # broadcast to correct shape if necessary
    if n_independent_part.shape != (n_sources, n_harmonics):
        n_independent_part = np.broadcast_to(n_independent_part.decompose(),
                                             (n_harmonics, n_sources)).T

    N, E = np.meshgrid(n, ecc)
    if interpolated_g is None:
        n_dependent_part = (peters_g(N, E) / N)**(1/2)
    else:
        g_vals = interpolated_g(n, ecc)

        # unsort the output array if there is more than one eccentricity
        if isinstance(ecc, (np.ndarray, list)) and len(ecc) > 1:
            g_vals = g_vals[np.argsort(ecc).argsort()]
        n_dependent_part = (g_vals / N)**(0.5) / N

    h_c = n_independent_part * n_dependent_part
    return h_c
