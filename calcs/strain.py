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

    n : `int`
        harmonic of the orbital frequency

    dist : `float/array`
        distance to the binary

    Returns
    -------
    h_0 : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """

    prefac = (2**(25/3) / 5)**(0.5) * c.G**(5/3) / c.c**4
    h_0 = prefac * m_c**(5/3) * (np.pi * f_orb)**(2/3) / dist *\
          peters_g(n, ecc)**(1/2) / n
    return h_0.decompose()


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

    n : `int`
        harmonic of the orbital frequency 

    dist : `float/array`
        distance to the binary

    t_obs : `float/array`
        observation duration

    Returns
    -------
    h_c : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """

    prefac = (2 / (3 * np.pi**(4/3)))**(0.5) * c.G**(5/6) / c.c**(3/2)
    h_c = prefac * m_c**(5/6) / dist * (n * f_orb)**(-1/6) \
            * (2 / n)**(1/3) * (peters_g(n, ecc) / peters_f(ecc))**(0.5)
    return h_c.decompose()
