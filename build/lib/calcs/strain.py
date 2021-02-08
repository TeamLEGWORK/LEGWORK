"""Computes several types of gravitational wave strains"""

from astropy import constants as c
from lisa_quick_calcs.utils import peters_g
import numpy as np

def h_0_n(m_c, f_orb, ecc, n, dist):
    """Computes the dimensionless strain of a general binary
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
        distance to the binary in units of kpc
    t_obs : `float/array`
        observation duration in units of seconds

    Returns
    -------
    h_0 : `float/array`
        dimensionless strain in the quadrupole approximation (unitless)
    """

    prefac = (128/5)**(0.5) * 2**(5/3) * c.G**(5/3) / c.c**4
    h_0 = prefac * m_c**(5/3) * (np.pi * f_orb)**(2/3) / dist *\
          peters_g(n, ecc)**(1/2) / n
    return h_0

