"""Computes several types of gravitational wave strains"""

import astropy.constants as c
from calcs.utils import peters_g, peters_f
import numpy as np


def h_0_n_2(m_c, f_orb, ecc, n, dist):
    """Computes the dimensionless power of a general binary
    radiating gravitational waves in the quadrupole approximation
    at the nth harmonic of the orbital frequency

    Params
    ------
    m_c : `float/array`
        chirp mass of the binary in units of kg

    f_orb : `float/array`
        orbital frequency in units of Hz

    ecc : `float/array`
        eccentricity

    n : `int`
        harmonic of the orbital frequency

    dist : `float/array`
        distance to the binary in units of meters

    Returns
    -------
    h_0**2 : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """

    prefac = (2**(25/3) / 5)**(0.5) * c.G**(5/3) / c.c**4
    h_0 = prefac * m_c**(5/3) * (np.pi * f_orb)**(2/3) / dist *\
          peters_g(n, ecc)**(1/2) / n
    return h_0**2


def h_c_n_2(m_c, f_orb, ecc, n, dist):
    """Computes the dimensionless characteristic power of a general
    binary radiating gravitational waves in the quadrupole approximation
    at the nth harmonic of the orbital frequency

    Params
    ------
    m_c : `float/array`
        chirp mass of the binary in units of kg

    f_orb : `float/array`
        orbital frequency in units of Hz

    ecc : `float/array`
        eccentricity

    n : `int`
        harmonic of the orbital frequency 

    dist : `float/array`
        distance to the binary in units ofmeters

    t_obs : `float/array`
        observation duration in units of seconds

    Returns
    -------
    h_c_2 : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """

    prefac = 2/(3*np.pi**(4/3)) * c.G**(5/3) / c.c**3
    h_c_2 = prefac * m_c**(5/3) / dist**2 * (n*f_orb)**(-1/3) *\
            (2/n)**(2/3) * peters_g(n, ecc) / peters_f(ecc)
    return h_c_2
