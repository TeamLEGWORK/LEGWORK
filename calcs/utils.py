"""`utils` for gw calcs"""

from scipy.special import jv
from astropy import constants as c
import numpy as np


def chirp_mass(m_1, m_2):
    """Computes chirp mass of a binary system

    Params
    ------
    m_1 : `float/array`
        more massive binary component

    m_2 : `float/array`
        less massive binary component

    Returns
    -------
    m_c : `float/array`
        chirp mass of the binary
    """

    m_c = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)
    return m_c


def peters_g(n, e):
    """relative power of gravitational radiation at nth harmonic
    from Peters and Mathews (1963)

    Params
    ------
    n : `int`
        harmonic of interest

    e : `array`
        eccentricity

    Returns
    -------
    g : `array`
        Fourier decomposition
    """

    bracket_1 = jv(n-2, n*e) - 2*e*jv(n-1, n*e) +\
        2/n*jv(n, n*e) + 2*e*jv(n+1, n*e) -\
        jv(n+2, n*e)
    bracket_2 = jv(n-2, n*e) - 2*jv(n, n*e) + jv(n+2, n*e)
    bracket_3 = jv(n, n*e)

    g = n**4/32 * (bracket_1**2 + (1-e**2) * bracket_2**2 +
                   4/(3*n**3) * bracket_3**2)

    return g


def peters_f(e):
    """integrated enhancement factor of gravitational radiation
    from an eccentric source from Peters and Mathews (1963)

    Params
    ------
    e : `array`
        eccentricity

    Returns
    -------
    f : `array`
        enhancement factor
    """

    numerator = 1 + (73/24)*e**2 + (37/96)*e**4
    denominator = (1 - e**2)**(7/2)

    f = numerator/denominator

    return f


def get_a_from_f_orb(f_orb, m_1, m_2):
    """Converts orbital frequency to separation using Kepler's
    third law all units are SI

    Params
    ------
    f_orb : `array`
        orbital frequency

    m_1 : `array`
        primary mass

    m_2 : `array`
        secondary mass

    Returns
    -------
    a : `array`
        separation
    """

    a = (c.G * (m_1 + m_2) / (2 * np.pi * f_orb)**2)**(1/3)

    return a


def get_f_orb_from_a(a, m_1, m_2):
    """Converts orbital frequency to separation using Kepler's
    third law where all units are SI

    Params
    ------
    a : `array`
        separation

    m_1 : `array`
        primary mass

    m_2 : `array`
        secondary mass

    Returns
    -------
    f_orb : `array`
        orbital frequency
    """

    f_orb = ((c.G * (m_1 + m_2) / a**3))**(0.5) / (2 * np.pi)

    return f_orb


def beta(m_1, m_2):
    """Computes the beta factor in Peters & Mathews calculations
    with all units in SI

    Params
    ------
    m_1 : `array`
        primary mass

    m_2 : `array`
        secondary mass

    Returns
    -------
    b : `array`
        beta factor in SI units
    """

    b = 64/5 * c.G**3/c.c**5 * m_1*m_2 * (m_1 + m_2)
    return b


def c_0(a_i, e_i):
    """Computes the c_0 factor in Peters and Mathews calculations

    Params
    ------
    a_i : `array`
        initial separation with astropy units

    e_i : `array`
        initial eccentricity

    Returns
    -------
    c0 : `array`
        c factor in SI units
    """

    c0 = a_i * (1-e_i**2) * e_i**(-12/19) *\
        (1 + (121/304)*e_i**2)**(-870/2299)
    return c0
