"""A collection of miscellaneous utility functions"""

from scipy.special import jv
from astropy import constants as c
from astropy import units as u
import numpy as np
import legwork.evol as evol

__all__ = ['chirp_mass', 'peters_g', 'peters_f', 'get_a_from_f_orb',
           'get_f_orb_from_a', 'beta', 'c_0', 'determine_stationarity',
           'fn_dot']


def chirp_mass(m_1, m_2):
    """Computes chirp mass of a binary system

    Parameters
    ----------
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

    Parameters
    ----------
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

    Parameters
    ----------
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

    Parameters
    ----------
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

    Parameters
    ----------
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

    Parameters
    ----------
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


def c_0(a_i, ecc_i):
    """Computes the c_0 factor in Peters and Mathews calculations

    Parameters
    ----------
    a_i : `array`
        initial separation with astropy units

    ecc_i : `array`
        initial eccentricity

    Returns
    -------
    c0 : `array`
        c factor in SI units
    """

    c0 = a_i * (1 - ecc_i**2) * ecc_i**(-12/19) \
        * (1 + (121/304)*ecc_i**2)**(-870/2299)
    return c0


def get_a_from_ecc(ecc, c_0):
    """Convert eccentricity to semi-major axis

    Use initial conditions and Peters (1964) Eq. 5.11 to convert ``ecc`` to
    ``a``.

    Parameters
    ----------
    ecc : `float/array`
        eccentricity

    c_0 : `float/array`
        peters c_0 constant, must have units of length
        (see :meth:`legwork.utils.c_0`)

    Returns
    -------
    a : `float/array`
        semi-major axis"""

    a = c_0 * ecc**(12/19) / (1 - ecc**2) \
        * (1 + (121/304) * ecc**2)**(870/2299)
    return a


def determine_stationarity(f_orb_i, t_evol, ecc_i,
                           m_1=None, m_2=None, m_c=None, stat_tol=1e-2):
    """Determine whether a binary is stationary by checking how
    much its orbital frequency changes over t_evol time

    This function provides a conservative estimate in that some
    binaries that are stationary may be marked as evolving. This
    is because the eccentricity also evolves but only use the
    initial value. Solving this in full would require the same
    amount of time as assuming the binary is evolving.

    Parameters
    ----------
    forb_i : `float/array`
        initial orbital frequency

    t_evol : `float`
        time over which the frequency evolves

    ecc : `float/array`
        initial eccentricity

    m_1 : `float/array`
        primary mass (required if `m_c` is None)

    m_2 : `float/array`
        secondary mass (required if `m_c` is None)

    m_c : `float/array`
        chirp mass (overrides `m_1` and `m_2`)

    stat_tol : `float`
        fractional change in frequency above which we do not
        consider a binary to be stationary

    Returns
    -------
    stationary : `bool/array`
        mask of whether each binary is stationary
    """
    # calculate chirp mass if necessary
    if m_c is None:
        if m_1 is None or m_1 is None:
            raise ValueError("`m_1` and `m_2` are required if `m_c` is None")
        m_c = chirp_mass(m_1, m_2)

    # calculate the final frequency
    f_orb_f = evol.evolve_f_orb_circ(f_orb_i=f_orb_i, m_c=m_c,
                                     t_evol=t_evol, ecc_i=ecc_i)

    # check the stationary criterion
    stationary = (f_orb_f - f_orb_i) / f_orb_i <= stat_tol
    return stationary


def fn_dot(m_c, f_orb, e, n):
    """Rate of change of nth frequency of a binary

    Parameters
    ----------
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


def ensure_array(*args):
    """convert arguments to numpy arrays

    Ignore any None values, convert any lists to numpy arrays, wrap any other
    types in lists and convert to numpy arrays
    """
    array_args = [None for i in range(len(args))]
    any_not_arrays = False
    for i in range(len(array_args)):
        exists = args[i] is not None
        has_units = isinstance(args[i], u.quantity.Quantity)
        if exists and has_units:
            if not isinstance(args[i].value, np.ndarray):
                any_not_arrays = True
                array_args[i] = np.asarray([args[i].value]) * args[i].unit
            else:
                array_args[i] = args[i]
        elif exists and not has_units:
            if not isinstance(args[i], np.ndarray):
                if not isinstance(args[i], list):
                    any_not_arrays = True
                    array_args[i] = np.asarray([args[i]])
                else:
                    array_args[i] = np.asarray(args[i])
            else:
                array_args[i] = args[i]
        else:
            array_args[i] = args[i]
    return array_args, any_not_arrays
