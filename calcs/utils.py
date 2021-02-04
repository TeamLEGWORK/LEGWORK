"""`utils` for gw calcs"""

from scipy.special import jv
from astropy import constants as c
from astropy import units as u
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

    bracket_1 = jv(n-2, n*e) - 2*e*jv(n-1, n*e) \
                + 2/n*jv(n, n*e) + 2*e*jv(n+1, n*e) \
                - jv(n+2, n*e)
    bracket_2 = jv(n-2, n*e) - 2*jv(n, n*e) + jv(n+2, n*e)
    bracket_3 = jv(n, n*e)

    g = n**4/32 * (bracket_1**2 + (1 - e**2) * bracket_2**2 +
                   4 / (3 * n**3) * bracket_3**2)

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
    return a.to(u.AU)


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

    b = 64/5 * c.G**3/c.c**5 * m_1 * m_2 * (m_1 + m_2)
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

    c0 = a_i * (1-e_i**2) * e_i**(-12/19) \
         * (1 + (121/304)*e_i**2)**(-870/2299)
    return c0


def determine_stationarity(m_1, m_2, forb_i, t_evol, ecc, stat_tol=1e-2):
    """Determine whether a binary is stationary by checking how
    much its orbital frequency changes over t_evol time

    This function provides a conservative estimate in that some
    binaries that are stationary may be marked as evolving. This
    is because the eccentricity also evolves but only use the
    initial value. Solving this in full would require the same
    amount of time as assuming the binary is evolving.

    Params
    ------
    m_1 : `float/array`
        primary mass

    m_2 : `float/array`
        secondary mass

    forb_i : `float/array`
        initial orbital frequency

    t_evol : `float`
        time over which the frequency evolves

    ecc : `float/array`
        initial eccentricity

    stat_tol : `float`
        fractional change in frequency above which we do not
        consider a binary to be stationary

    Returns
    -------
    stationary : `bool/array`
        mask of whether each binary is stationary
    """
    m_c = chirp_mass(m_1, m_2)
    # calculate the inner part of the final frequency equation
    inner_part = forb_i**(-8/3) - 2**(32/3) * np.pi**(8/3) \
                 * t_evol / (5 * c.c**5) * (c.G * m_c)**(5/3) * peters_f(ecc)

    # any merged binaries will have a negative inner part
    inspiral = inner_part >= 0.0

    # calculate the change in frequency (set to 10^10 Hz if merged)
    delta_f = np.repeat(1e10, len(forb_i)) * u.Hz
    delta_f[inspiral] = np.power(inner_part[inspiral], -3/8) - forb_i[inspiral]

    stationary = delta_f / forb_i <= stat_tol

    return stationary


def fn_dot(m_c, f_orb, e, n):
    """Rate of change of nth frequency of a binary

    Params
    ------
    m_c : `float/array`
        chirp mass

    f_orb : `float/array`
        orbital frequency

    e : `float/array`
        eccentricity

    n : `int`
        harmonic of interest

    Returns
    -------
    fn_dot : `float/array`
        rate of change of nth frequency
    """
    fn_dot = (48 * n) / (5 * np.pi) * (c.G * m_c)**(5/3) / c.c**5 \
             * (2 * np.pi * f_orb)**(11/3) * peters_f(e)
    return fn_dot.to(u.Hz / u.yr)
