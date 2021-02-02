"""Computes several types of gravitational wave strains"""

import astropy.constants as c
from calcs.utils import peters_g, peters_f
import numpy as np


def h_0_n(m_c, f_orb, ecc, n, dist):
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

    Returns
    -------
    h_0 : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
        shape of array is `(number of sources, number of harmonics)`
    """
    # calculate how many harmonics and sources
    n_harmonics = 1 if isinstance(n, (int, np.int64, np.int)) else len(n)
    n_sources = len(m_c) if isinstance(m_c.value, (list, np.ndarray)) else 1

    # work out strain for n independent part and broadcast to correct shape
    prefac = (2**(28/3) / 5)**(0.5) * c.G**(5/3) / c.c**4
    n_independent_part = prefac * m_c**(5/3) * (np.pi * f_orb)**(2/3) / dist
    n_independent_part = np.broadcast_to(n_independent_part.decompose(), (n_harmonics, n_sources)).T

    N, E = np.meshgrid(n, ecc)
    n_dependent_part = peters_g(N, E)**(1/2) / N

    h_0 = n_independent_part * n_dependent_part
    return h_0


def h_c_n(m_c, f_orb, ecc, n, dist):
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

    Returns
    -------
    h_c : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """
    # calculate how many harmonics and sources
    n_harmonics = 1 if isinstance(n, (int, np.int64, np.int)) else len(n)
    n_sources = len(m_c) if isinstance(m_c.value, (list, np.ndarray)) else 1

    # work out strain for n independent part and broadcast to correct shape
    prefac = (2**(5/3) / (3 * np.pi**(4/3)))**(0.5) * c.G**(5/6) / c.c**(3/2)
    n_independent_part = prefac * m_c**(5/6) / dist * f_orb**(-1/6) / peters_f(ecc)**(0.5)
    n_independent_part = np.broadcast_to(n_independent_part.decompose(), (n_harmonics, n_sources)).T

    N, E = np.meshgrid(n, ecc)
    n_dependent_part = (peters_g(N, E) / N)**(1/2)

    h_c = n_independent_part * n_dependent_part
    return h_c
