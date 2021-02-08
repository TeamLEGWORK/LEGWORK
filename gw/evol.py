"""`evolve with peters!`"""

import gw.utils as utils
from numba import jit
from scipy.integrate import odeint
import numpy as np
import astropy.units as u


@jit
def de_dt(e, times, beta, c_0):
    """Computes the evolution of the eccentricity from the emission
    of gravitational waves following Peters & Mathews 1964

    Params
    ------
    e : `float`
        initial eccentricity

    times : `array`
        evolution times in SI units

    beta : `float`
        factor defined in Peters and Mathews 1964 in SI

    c_0 : `float`
        factor defined in Peters and Mathews 1964 in SI


    Returns
    -------
    dedt : `array`
        eccentricity evolution
    """
    dedt = -19/12 * beta/c_0**4 * (e**(29/19)*(1 - e**2)**(3/2)) \
                                / (1+(121/304)*e**2)**(1181/2299)

    return dedt


def get_a_evol(a_i, e_evol, beta, c_0, times):
    """Calculates the separation evolution of a binary following
    Peters 1964


    Params
    ------
    a_i : `float`
        initial separation in SI units

    e_i : `float`
        initial eccentricity

    e_evol : `float`
        eccentricity evolution

    beta : `float`
        factor defined in Peters and Mathews 1964 in SI

    c_0 : `float`
        factor defined in Peters and Mathews 1964 in SI

    times : `array`
        array of times for evolution in SI units

    Returns
    -------
    a_evol : `float`
        separation evolution for eccentricity evolution in SI
    """

    if e_evol.all() == 0.0:
        difference = a_i**4 - 4*beta*times
        difference = np.where(difference.value <= 0.0, 0.0, difference)
        a_evol = difference**(1/4)
    else:
        term_1 = c_0 * e_evol**(12/19) / (1-e_evol**2)
        term_2 = (1 + (121/304)*e_evol**2)**(870/2299)
        a_evol = term_1 * term_2

    return a_evol


def get_e_evol(beta, c_0, ecc_i, times):
    """Calculates the eccentricity evolution of a binary
    with beta and c_0 factors and evolution times following
    Peters 1964

    Params
    ------
    beta : `float`
        Peters beta parameter in SI units, calculated in utils

    c_0 : `float`
        Peters c_0 parameter in SI units, calculated in utils

    ecc_i : `float`
        initial eccentricity

    times : `array`
        array of times for evolution in SI units

    Returns
    -------
    e_evol : `array`
        array of eccentricities evolved for the times provided
    """

    e_evol = odeint(de_dt, ecc_i, times, args=(beta.value, c_0.value))

    return e_evol.flatten()


def get_f_and_e(m_1, m_2, f_orb_i, e_i, t_evol, n_step):
    """Evolves a binary due to the emission of gravitational waves
    and returns the final separation at t_evol

    Params
    ------
    m_1 : `float/array`
        more massive binary component in units of kg

    m_2 : `float/array`
        less massive binary component in units of kg

    f_orb_i : `float/array`
        initial orbital frequency in units of Hz

    e_i : `float/array`
        initial eccentricity

    t_evol : `float`
        evolution duration in units of sec

    n_step : `int`
        number of steps to take in evolution

    Returns
    -------
    f_orb_evol : `array`
        frequency evolution for n_step times up to t_evol

    e_evol : `array`
        eccentricity evolution for n_step times up to t_evol
    """

    a_i = utils.get_a_from_f_orb(f_orb=f_orb_i, m_1=m_1, m_2=m_2)
    beta = utils.beta(m_1, m_2)
    times = np.linspace(0, t_evol, n_step)

    # Only treat single eccentric sources, so any cases where
    # len(e_i) > 1 are circular.
    if type(e_i) != np.float64:
        c_0 = 0.0
        # treat as circular
        e_evol = np.zeros_like(n_step)

    else:
        c_0 = utils.c_0(a_i, e_i)
        # treat as eccentric but since we have to integrate one by one,
        # we call get_f_e for a binary one by one
        e_evol = get_e_evol(beta=beta, c_0=c_0, ecc_i=e_i, times=times)

    a_evol = get_a_evol(a_i=a_i, e_evol=e_evol,
                        beta=beta, c_0=c_0, times=times)

    # change merged binaries to extremely small separations
    a_evol = np.where(a_evol.value == 0.0, 1e-30 * a_evol.unit, a_evol)
    f_orb_evol = utils.get_f_orb_from_a(a=a_evol, m_1=m_1, m_2=m_2)

    # change frequencies back to 1Hz since LISA can't measure above
    f_orb_evol = np.where(a_evol.value == 1e-30, 1 * u.Hz, f_orb_evol)
    return f_orb_evol.to(u.Hz), e_evol


def get_t_merge_circ(m_1, m_2, f_orb_i):
    """Computes the merger time in seconds for a circular binary
    from Peters 1964

    Params
    ------
    m_1 : `array`
        primary mass

    m_2 : `array`
        secondary mass

    f_orb_i : `array`
        initial orbital frequency

    Returns
    -------
    t_merge : `array`
        merger time
    """

    a_i = utils.get_a_from_f_orb(f_orb=f_orb_i, m_1=m_1, m_2=m_2)
    beta = utils.beta(m_1, m_2)

    t_merge = a_i**4 / (4*beta)

    return t_merge


def get_t_merge_ecc(m_1, m_2, f_orb_i, ecc_i):
    """Computes the merger time in seconds for a circular binary
    from Peters 1964

    Params
    ------
    m_1 : `array`
        primary mass

    m_2 : `array`
        secondary mass

    f_orb_i : `array`
        initial orbital frequency

    ecc_i : `array`
        initial eccentricity

    Returns
    -------
    t_merge : `array`
        merger time
    """

    a_i = utils.get_a_from_f_orb(f_orb=f_orb_i, m_1=m_1, m_2=m_2)
    beta = utils.beta(m_1, m_2)

    t_merge = a_i**4 / (4*beta)

    return t_merge.to(u.Gyr)
