"""A collection of miscellaneous utility functions"""

from scipy.special import jv
from astropy import constants as c
from astropy import units as u
import numpy as np

__all__ = ['chirp_mass', 'peters_g', 'peters_f', 'get_a_from_f_orb', 'get_f_orb_from_a', 'get_a_from_ecc',
           'beta', 'c_0', 'fn_dot', 'ensure_array', 'D_plus_squared',
           'D_cross_squared', 'D_plus_D_cross', 'F_plus_squared', 'F_cross_squared']


def chirp_mass(m_1, m_2):
    """Computes chirp mass of binaries

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    Returns
    -------
    m_c : `float/array`
        Chirp mass
    """
    m_c = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)

    # simplify units if present
    if isinstance(m_c, u.quantity.Quantity):
        m_c = m_c.to(u.Msun)

    return m_c


def peters_g(n, e):
    """Compute g(n, e) from Peters and Mathews (1963) Eq.20

    This function gives the relative power of gravitational radiation at the nth harmonic

    Parameters
    ----------
    n : `int/array`
        Harmonic(s) of interest

    e : `float/array`
        Eccentricity

    Returns
    -------
    g : `array`
        g(n, e) from Peters and Mathews (1963) Eq. 20
    """

    bracket_1 = jv(n-2, n*e) - 2*e*jv(n-1, n*e) + 2/n*jv(n, n*e) + 2*e*jv(n+1, n*e) - jv(n+2, n*e)
    bracket_2 = jv(n-2, n*e) - 2*jv(n, n*e) + jv(n+2, n*e)
    bracket_3 = jv(n, n*e)

    g = n**4/32 * (bracket_1**2 + (1 - e**2) * bracket_2**2 + 4 / (3 * n**3) * bracket_3**2)

    return g


def peters_f(e):
    """f(e) from Peters and Mathews (1963) Eq.17

    This function gives the integrated enhancement factor of gravitational radiation from an eccentric
    source compared to an equivalent circular source.

    Parameters
    ----------
    e : `float/array`
        Eccentricity

    Returns
    -------
    f : `float/array`
        Enhancement factor

    Notes
    -----
    Note that this function represents an infinite sum of g(n, e) - :meth:`legwork.utils.peters_g`
    """

    numerator = 1 + (73/24)*e**2 + (37/96)*e**4
    denominator = (1 - e**2)**(7/2)

    f = numerator / denominator

    return f


def get_a_from_f_orb(f_orb, m_1, m_2):
    """Converts orbital frequency to semi-major axis

    Using Kepler's third law, convert orbital frequency to semi-major axis.
    Inverse of :func:`legwork.utils.get_f_orb_from_a`.

    Parameters
    ----------
    f_orb : `float/array`
        Orbital frequency

    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    Returns
    -------
    a : `float/array`
        Semi-major axis
    """
    a = (c.G * (m_1 + m_2) / (2 * np.pi * f_orb)**2)**(1/3)

    # simplify units if present
    if isinstance(a, u.quantity.Quantity):
        a = a.to(u.AU)

    return a


def get_f_orb_from_a(a, m_1, m_2):
    """Converts semi-major axis to orbital frequency

    Using Kepler's third law, convert semi-major axis to orbital frequency.
    Inverse of :func:`legwork.utils.get_a_from_f_orb`.

    Parameters
    ----------
    a : `float/array`
        Semi-major axis

    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    Returns
    -------
    f_orb : `float/array`
        Orbital frequency
    """
    f_orb = ((c.G * (m_1 + m_2) / a**3))**(0.5) / (2 * np.pi)

    # simplify units if present
    if isinstance(f_orb, u.quantity.Quantity):
        f_orb = f_orb.to(u.Hz)

    return f_orb


def beta(m_1, m_2):
    """Compute beta defined in Peters and Mathews (1964) Eq.5.9

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    Returns
    -------
    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq.5.9.
    """
    beta = 64 / 5 * c.G**3 / c.c**5 * m_1 * m_2 * (m_1 + m_2)

    # simplify units if present
    if isinstance(beta, u.quantity.Quantity):
        beta = beta.to(u.m**4 / u.s)

    return beta


def c_0(a_i, ecc_i):
    """Computes the c_0 factor in Peters and Mathews (1964) Eq.5.11

    Parameters
    ----------
    a_i : `float/array`
        Initial semi-major axis

    ecc_i : `float/array`
        Initial eccentricity

    Returns
    -------
    c_0 : `float`
        Constant defined in Peters and Mathews (1964) Eq.5.11
    """
    c_0 = a_i * (1 - ecc_i**2) * ecc_i**(-12/19) * (1 + (121/304)*ecc_i**2)**(-870/2299)

    # simplify units if present
    if isinstance(c_0, u.quantity.Quantity):
        c_0 = c_0.to(u.AU)

    return c_0


def get_a_from_ecc(ecc, c_0):
    """Convert eccentricity to semi-major axis

    Use initial conditions and Peters (1964) Eq. 5.11 to convert ``ecc`` to ``a``.

    Parameters
    ----------
    ecc : `float/array`
        Eccentricity

    c_0 : `float`
        Constant defined in Peters and Mathews (1964) Eq. 5.11. See :meth:`legwork.utils.c_0`

    Returns
    -------
    a : `float/array`
        Semi-major axis"""

    a = c_0 * ecc**(12/19) / (1 - ecc**2) * (1 + (121/304) * ecc**2)**(870/2299)

    # simplify units if present
    if isinstance(a, u.quantity.Quantity):
        a = a.to(u.AU)

    return a


def fn_dot(m_c, f_orb, e, n):
    """Rate of change of nth frequency of a binary

    Parameters
    ----------
    m_c : `float/array`
        Chirp mass

    f_orb : `float/array`
        Orbital frequency

    e : `float/array`
        Eccentricity

    n : `int`
        Harmonic of interest

    Returns
    -------
    fn_dot : `float/array`
        Rate of change of nth frequency
    """
    fn_dot = (48 * n) / (5 * np.pi) * (c.G * m_c)**(5/3) / c.c**5 * (2 * np.pi * f_orb)**(11/3) * peters_f(e)

    # simplify units if present
    if isinstance(fn_dot, u.quantity.Quantity):
        fn_dot = fn_dot.to(u.Hz / u.yr)

    return fn_dot


def ensure_array(*args):
    """Convert arguments to numpy arrays

    Convert arguments based on the following rules

        - Ignore any None values
        - Convert any lists to numpy arrays
        - Wrap any other types in lists and convert to numpy arrays

    Parameters
    ----------
    args : `any`
        Supply any number of arguments of any type

    Returns
    -------
    array_args : `any`
        Args converted to numpy arrays

    any_not_arrays : `bool`
        Whether any arg is not a list or None or a numpy array
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


def D_plus_squared(theta, phi):
    """Required for the detector responses <F_+^2>, <F_x^2>, <F_+F_x>

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    Returns
    -------
    D_plus_2 : `float/array`
        factor used for response; see eq. 44 of Cornish and Larson (2003)
    """

    term_1 = 158 * np.cos(theta)**2
    term_2 = 7 * np.cos(theta)**4
    term_3 = -162 * np.sin(2 * phi)**2 * (1 + np.cos(theta)**2)**2
    D_plus_2 = (3 / 2048) * (487 + term_1 + term_2 + term_3)

    return D_plus_2


def D_cross_squared(theta, phi):
    """Required for the detector responses <F_+^2>, <F_x^2>, <F_+F_x>

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    Returns
    -------
    D_cross_2 : `float/array`
        factor used for response; see eq. 44 of Cornish and Larson (2003)
    """

    term_1 = 120 * np.sin(theta)**2
    term_2 = np.cos(theta)**2
    term_3 = 162 * np.sin(2 * phi)**2 * np.cos(theta)**2

    D_cross_2 = (3 / 512) * (term_1 + term_2 + term_3)

    return D_cross_2


def D_plus_D_cross(theta, phi):
    """Required for the detector responses <F_+^2>, <F_x^2>, <F_+F_x>

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    Returns
    -------
    D_plus_cross : `float/array`
        factor used for response; see eq. 44 of Cornish and Larson (2003)
    """

    term_1 = np.cos(theta) * np.sin(2 * phi)
    term_2 = 2 * np.cos(phi)**2 - 1
    term_3 = 1 + np.cos(theta)**2

    D_plus_cross = (243 / 512) * term_1 * term_2 * term_3

    return D_plus_cross


def F_plus_squared(theta, phi, psi):
    """Compute the auto-correlated detector response for the plus polarization

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    psi : `float/array`
        polarization of the source

    Returns
    -------
    F_plus_2 : `float/array`
        Auto-correlated detector response for the plus polarization
    """

    term_1 = np.cos(2 * psi)**2 * D_plus_squared(theta, phi)
    term_2 = -np.sin(4 * psi) * D_plus_D_cross(theta, phi)
    term_3 = np.sin(2 * psi)**2 * D_cross_squared(theta, phi)

    F_plus_2 = (1 / 4) * (term_1 + term_2 + term_3)

    return F_plus_2


def F_cross_squared(theta, phi, psi):
    """Compute the auto-correlated detector response for the cross polarization

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    psi : `float/array`
        polarization of the source

    Returns
    -------
    F_cross_2 : `float/array`
        Auto-correlated detector response for the cross polarization
    """

    term_1 = np.cos(2 * psi)**2 * D_cross_squared(theta, phi)
    term_2 = np.sin(4 * psi) * D_plus_D_cross(theta, phi)
    term_3 = np.sin(2 * psi)**2 * D_plus_squared(theta, phi)

    F_cross_2 = (1 / 4) * (term_1 + term_2 + term_3)

    return F_cross_2
