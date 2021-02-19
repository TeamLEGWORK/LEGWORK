"""Functions using equations from Peters (1964) to calculate inspiral times and
evolve parameters."""

import legwork.utils as utils
from numba import jit
from scipy.integrate import odeint, quad
import numpy as np
import astropy.units as u
import astropy.constants as c

__all__ = ['de_dt', 'evol_circ', 'evol_ecc',
           'get_t_merge_circ', 'get_t_merge_ecc', 'evolve_f_orb_circ',
           'check_mass_freq_input', 'create_timesteps_array',]


@jit
def de_dt(e, times, beta, c_0):                             # pragma: no cover
    """Computes the evolution of the eccentricity from the emission
    of gravitational waves following Peters & Mathews 1964

    Parameters
    ----------
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
    dedt = -19 / 12 * beta / c_0**4 * (e**(-29 / 19) * (1 - e**2)**(3/2)) \
        / (1 + (121/304) * e**2)**(1181/2299)
    return dedt


def check_mass_freq_input(beta=None, m_1=None, m_2=None,
                          a_i=None, f_orb_i=None):
    """check that mass and frequency input is valid

    Parameters
    ----------

    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9 (if supplied ``m_1`` and
        `m_2` are ignored)

    m_1 : `float/array`
        primary mass (required if ``beta`` is None)

    m_2 : `float/array`
        secondary mass (required if ``beta`` is None)

    a_i : `float/array`
        initial semi major axis (if supplied ``f_orb_i`` is ignored)

    f_orb_i : `float/array`
        initial orbital frequency (required if ``a_i`` is None)

    Returns
    -------
    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9

    a_i : `float/array`
        initial semi major axis
    """
    # ensure that beta is supplied or calculated
    if beta is None and (m_1 is None or m_2 is None):
        raise ValueError("Either `beta` or (`m_1`, `m_2`) must be supplied")
    elif beta is None:
        beta = utils.beta(m_1, m_2)

    # ensure that a_i is supplied or calculated
    if a_i is None and f_orb_i is None:
        raise ValueError("Either `a_i` or `f_orb_i` must be supplied")
    elif a_i is None and (m_1 is None or m_2 is None):
        raise ValueError("Individual masses `m_1` and `m_2` are required \
                         if no value of `a_i` is supplied")
    elif a_i is None:
        a_i = utils.get_a_from_f_orb(f_orb=f_orb_i, m_1=m_1, m_2=m_2)
    return beta, a_i


def create_timesteps_array(a_i, beta, ecc_i=None,
                           t_evol=None, n_step=100, timesteps=None):
    """Create an array of timesteps

    Parameters
    ----------
    t_evol : `float/array`
        amount of time for which to evolve each binaries. Required if
        ``timesteps`` is None. Defaults to merger times.

    n_steps : `int`
        Number of timesteps to take between t=0 and t=``t_evol``. Required if
        ``timesteps`` is None. Defaults to 100.

    timesteps : `float/array`
        Array of exact timesteps to take when evolving each binary. Must be
        monotonically increasing and start with t=0. Either supply a 1D array
        to use for every binary or a 2D array that has a different array of
        timesteps for each binary. ``timesteps`` is used in place of
        ``t_evol`` and ``n_steps`` and takes precedence over them.

    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9

    a_i : `float/array`
        initial semi major axis

    Returns
    -------
    timesteps : `float/array`
        array of timesteps for each binary
    """
    # create timesteps array if not provided
    if timesteps is None:
        # if not evolution times given use merger times
        if t_evol is None:
            t_evol = get_t_merge_ecc(ecc_i=ecc_i, a_i=a_i, beta=beta)
        timesteps = np.linspace(0 * u.s, t_evol, n_step).T
    # broadcast the times to every source if only one array provided
    elif np.ndim(timesteps) == 1:
        timesteps = timesteps[:, np.newaxis]
    return timesteps


def evol_circ(t_evol=None, n_step=100, timesteps=None, beta=None, m_1=None,
              m_2=None, a_i=None, f_orb_i=None, output_vars='f_orb'):
    """Evolve an array of circular binaries for ``t_evol`` time

    Parameters
    ----------

    t_evol : `float/array`
        amount of time for which to evolve each binaries. Required if
        ``timesteps`` is None. Defaults to merger times.

    n_steps : `int`
        Number of timesteps to take between t=0 and t=``t_evol``. Required if
        ``timesteps`` is None. Defaults to 100.

    timesteps : `float/array`
        Array of exact timesteps to take when evolving each binary. Must be
        monotonically increasing and start with t=0. Either supply a 1D array
        to use for every binary or a 2D array that has a different array of
        timesteps for each binary. ``timesteps`` is used in place of
        ``t_evol`` and ``n_steps`` and takes precedence over them.

    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9 (if supplied ``m_1`` and
        `m_2` are ignored)

    m_1 : `float/array`
        primary mass (required if ``beta`` is None or if ``output_vars``
        contains a frequency)

    m_2 : `float/array`
        secondary mass (required if ``beta`` is None or if ``output_vars``
        contains a frequency)

    a_i : `float/array`
        initial semi major axis (if supplied ``f_orb_i`` is ignored)

    f_orb_i : `float/array`
        initial orbital frequency (required if ``a_i`` is None)

    output_vars : `str/array`
        list of **ordered** output vars, or a single var. Choose from any of
        ``a``, ``f_orb`` and ``f_GW`` for which of semi-major axis and
        orbital/GW frequency that you want. Default is ``f_orb``.

    Returns
    -------
    evolution : `array`
        array possibly containing semi-major axis and frequency
        at each timestep. Content determined by ``output_vars``.
    """
    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2,
                                      a_i=a_i, f_orb_i=f_orb_i)

    # convert output_vars to array if only str provided
    if isinstance(output_vars, str):
        output_vars = [output_vars]

    if len(set(["f_orb", "f_GW"])
           & set(output_vars)) > 0 and (m_1 is None or m_2 is None):
        raise ValueError("`m_1`` and `m_2` required if `output_vars` " +
                         "contains a frequency")
    timesteps = create_timesteps_array(a_i=a_i, beta=beta,
                                       ecc_i=np.zeros_like(a_i), t_evol=t_evol,
                                       n_step=n_step, timesteps=timesteps)

    # perform the evolution
    difference = a_i[:, np.newaxis]**4 - 4 * beta[:, np.newaxis] * timesteps
    difference = np.where(difference.value <= 0.0, 0.0, difference)
    a_evol = difference**(1/4)

    # calculate f_orb_evol if any frequency requested
    if len(set(["f_orb", "f_GW"]) & set(output_vars)) > 0:
        # change merged binaries to extremely small separations
        a_not0 = np.where(a_evol.value == 0.0, 1e-30 * a_evol.unit, a_evol)
        f_orb_evol = utils.get_f_orb_from_a(a=a_not0,
                                            m_1=m_1[:, np.newaxis],
                                            m_2=m_2[:, np.newaxis])

        # change frequencies back to 1Hz since LISA can't measure above
        f_orb_evol = np.where(a_not0.value == 1e-30, 1 * u.Hz, f_orb_evol)

    # construct evolution output
    evolution = []
    for var in output_vars:
        if var == "a":
            evolution.append(a_evol.to(u.AU))
        elif var == "f_orb":
            evolution.append(f_orb_evol.to(u.Hz))
        elif var == "f_GW":
            evolution.append(2 * f_orb_evol.to(u.Hz))
    return evolution if len(evolution) > 1 else evolution[0]


def evol_ecc(ecc_i, t_evol=None, n_step=100, timesteps=None, beta=None,
             m_1=None, m_2=None, a_i=None, f_orb_i=None,
             output_vars=['ecc', 'f_orb']):
    """Evolve an array of eccentric binaries for ``t_evol`` time

    Parameters
    ----------
    ecc_i : `float/array`
        initial eccentricity

    t_evol : `float/array`
        amount of time for which to evolve each binaries. Required if
        ``timesteps`` is None. Defaults to merger times.

    n_steps : `int`
        Number of timesteps to take between t=0 and t=``t_evol``. Required if
        ``timesteps`` is None. Defaults to 100.

    timesteps : `float/array`
        Array of exact timesteps to take when evolving each binary. Must be
        monotonically increasing and start with t=0. Either supply a 1D array
        to use for every binary or a 2D array that has a different array of
        timesteps for each binary. ``timesteps`` is used in place of
        ``t_evol`` and ``n_steps`` and takes precedence over them.

    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9 (if supplied ``m_1`` and
        `m_2` are ignored)

    m_1 : `float/array`
        primary mass (required if ``beta`` is None or if ``output_vars``
        contains a frequency)

    m_2 : `float/array`
        secondary mass (required if ``beta`` is None or if ``output_vars``
        contains a frequency)

    a_i : `float/array`
        initial semi major axis (if supplied ``f_orb_i`` is ignored)

    f_orb_i : `float/array`
        initial orbital frequency (required if ``a_i`` is None)

    output_vars : `array`
        list of **ordered** output vars, choose from any of ``ecc``, ``a``,
        ``f_orb`` and ``f_GW`` for which of eccentricty, semi-major axis and
        orbital/GW frequency that you want. Default is [``ecc``, ``f_orb``]

    Returns
    -------
    evolution : `tuple`
        tuple possibly containing eccentricty, semi-major axis and frequency
        at each timestep. Content determined by ``output_vars``
    """
    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2,
                                      a_i=a_i, f_orb_i=f_orb_i)

    # convert output_vars to array if only str provided
    if isinstance(output_vars, str):
        output_vars = [output_vars]

    if len(set(["f_orb", "f_GW"])
           & set(output_vars)) > 0 and (m_1 is None or m_2 is None):
        raise ValueError("`m_1`` and `m_2` required if `output_vars` " +
                         "contains a frequency")
    c_0 = utils.c_0(a_i=a_i, ecc_i=ecc_i)
    timesteps = create_timesteps_array(a_i=a_i, beta=beta, ecc_i=ecc_i,
                                       t_evol=t_evol, n_step=n_step,
                                       timesteps=timesteps)

    # get rid of the units for faster integration
    c_0 = c_0.to(u.m).value
    beta = beta.to(u.m**4 / u.s).value
    timesteps = timesteps.to(u.s).value

    # perform the evolution
    ecc_evol = np.array([odeint(de_dt, ecc_i[i], timesteps[i],
                                args=(beta[i], c_0[i])).flatten()
                         for i in range(len(ecc_i))])
    c_0 = c_0[:, np.newaxis] * u.m
    ecc_evol = np.nan_to_num(ecc_evol, nan=0.0)

    # calculate a_evol if any frequency or separation requested
    if len(set(["a", "f_orb", "f_GW"]) & set(output_vars)) > 0:
        a_evol = utils.get_a_from_ecc(ecc_evol, c_0)

        # calculate f_orb_evol if any frequency requested
        if len(set(["f_orb", "f_GW"]) & set(output_vars)) > 0:
            # change merged binaries to extremely small separations
            a_not0 = np.where(a_evol.value == 0.0, 1e-30 * a_evol.unit, a_evol)
            f_orb_evol = utils.get_f_orb_from_a(a=a_not0,
                                                m_1=m_1[:, np.newaxis],
                                                m_2=m_2[:, np.newaxis])

            # change frequencies back to 1Hz since LISA can't measure above
            f_orb_evol = np.where(a_not0.value == 1e-30, 1 * u.Hz, f_orb_evol)

    # construct evolution output
    evolution = []
    for var in output_vars:
        if var == "ecc":
            evolution.append(ecc_evol)
        elif var == "a":
            evolution.append(a_evol.to(u.AU))
        elif var == "f_orb":
            evolution.append(f_orb_evol.to(u.Hz))
        elif var == "f_GW":
            evolution.append(2 * f_orb_evol.to(u.Hz))
    return evolution if len(evolution) > 1 else evolution[0]


def get_t_merge_circ(beta=None, m_1=None, m_2=None,
                     a_i=None, f_orb_i=None):
    """Computes the merger time for a circular binary using Peters 1964

    Parameters
    ----------
    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9 (if supplied `m_1` and
        `m_2` are ignored)

    m_1 : `float/array`
        primary mass (required if `beta` is None)

    m_2 : `float/array`
        secondary mass (required if `beta` is None)

    a_i : `float/array`
        initial semi major axis (if supplied `f_orb_i` is ignored)

    f_orb_i : `float/array`
        initial orbital frequency (required if `a_i` is None)

    Returns
    -------
    t_merge : `float/array`
        merger time
    """
    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2,
                                      a_i=a_i, f_orb_i=f_orb_i)

    # apply Peters 1964 Eq. 5.9
    t_merge = a_i**4 / (4 * beta)

    return t_merge.to(u.Gyr)


def get_t_merge_ecc(ecc_i, a_i=None, f_orb_i=None,
                    beta=None, m_1=None, m_2=None,
                    small_e_tol=1e-2, large_e_tol=1 - 1e-2):
    """Computes the merger time for a binary using Peters 1964.
    We use one of Eq. 5.10, 5.14 or the two unlabelled equations
    after 5.14 in Peters 1964 depending on the eccentricity of
    the binary.

    Parameters
    ----------
    ecc_i : `float/array`
        initial eccentricity (if `ecc_i` is known to be 0.0 then use
        `get_t_merge_circ` instead)

    a_i : `float/array`
        initial semi major axis (if supplied `f_orb_i` is ignored)

    f_orb_i : `float/array`
        initial orbital frequency (required if `a_i` is None)

    beta : `float/array`
        beta(m_1, m_2) from Peters 1964 Eq. 5.9 (if supplied `m_1` and
        `m_2` are ignored)

    m_1 : `float/array`
        primary mass (required if `beta` is None)

    m_2 : `float/array`
        secondary mass (required if `beta` is None)

    small_e_tol : `float`
        eccentricity below which to apply the small e approximation
        (first unlabelled equation following Eq. 5.14 of Peters 1964)

    large_e_tol : `float`
        eccentricity above which to apply the large e approximation
        (second unlabelled equation following Eq. 5.14 of Peters 1964)

    Returns
    -------
    t_merge : `float/array`
        merger time
    """
    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2,
                                      a_i=a_i, f_orb_i=f_orb_i)

    # shortcut if all binaries are circular
    if np.all(ecc_i == 0.0):
        return get_t_merge_circ(beta=beta, a_i=a_i)

    # calculate c0 from Peters Eq. 5.11
    c0 = utils.c_0(a_i, ecc_i)

    @jit(nopython=True)
    def peters_5_14(e):                                 # pragma: no cover
        """ merger time from Peters Eq. 5.14 """
        return np.power(e, 29/19) * np.power(1 + (121/304)*e**2, 1181/2299) \
            / np.power(1 - e**2, 3/2)

    # case with array of binaries
    if isinstance(ecc_i, (np.ndarray, list)):
        # mask eccentricity based on tolerances
        circular = ecc_i == 0.0
        small_e = np.logical_and(ecc_i > 0.0, ecc_i < small_e_tol)
        large_e = ecc_i > large_e_tol
        other_e = np.logical_and(ecc_i >= small_e_tol, ecc_i <= large_e_tol)

        t_merge = np.zeros(len(ecc_i)) * u.Gyr

        # merger time for circular binaries (Peters Eq. 5.9)
        t_merge[circular] = a_i[circular]**4 / (4 * beta[circular])

        # merger time for low e binaries (Eq after Peters Eq. 5.14)
        t_merge[small_e] = c0[small_e]**4 / (4 * beta[small_e]) \
            * ecc_i[small_e]**(48/19)

        # merger time for high e binaries (2nd Eq after Peters Eq. 5.14)
        t_merge[large_e] = c0[large_e]**4 / (4 * beta[large_e]) \
            * ecc_i[large_e]**(48/19) * (768 / 425) \
            * (1 - ecc_i[large_e]**2)**(-1/2) \
            * (1 + 121/304 * ecc_i[large_e]**2)**(3480/2299)

        # merger time for general binaries (Peters Eq. 5.14)
        prefac = ((12 / 19) * c0[other_e]**4 / beta[other_e]).to(u.Gyr)
        t_merge[other_e] = prefac * [quad(peters_5_14, 0, ecc_i[other_e][i])[0]
                                     for i in range(len(ecc_i[other_e]))]
    # case with only one binary
    else:
        # conditions as above (no need for ecc=0.0 since it never reaches here)
        if ecc_i < small_e_tol:
            t_merge = c0**4 / (4 * beta) * ecc_i**(48/19)
        elif ecc_i > large_e_tol:
            t_merge = c0**4 / (4 * beta) * ecc_i**(48/19) * (768 / 425) \
                * (1 - ecc_i**2)**(-1/2) \
                * (1 + 121/304 * ecc_i**2)**(3480/2299)
        else:
            t_merge = ((12 / 19) * c0**4 / beta
                       * quad(peters_5_14, 0, ecc_i)[0])
    return t_merge.to(u.Gyr)


def evolve_f_orb_circ(f_orb_i, m_c, t_evol, ecc_i=0.0, merge_f=1e9 * u.Hz):
    """Evolve orbital frequency for `t_evol` time. This gives the exact final
    frequency for circular binaries. However, it will overestimate the final
    frequency for an eccentric binary and if an exact value is required then
    `evol.get_f_and_e()` should be used instead.

    Parameters
    ----------
    f_orb_i : `float/array`
        initial orbital frequency

    m_c : `float/array`
        chirp mass

    t_evol : `float`
        time over which the frequency evolves

    ecc_i : `float/array`
        initial eccentricity

    merge_f : `float`
        frequency to assign if the binary has already merged after `t_evol`

    Returns
    -------
    f_orb_f : `bool/array`
        final orbital frequency
    """
    # fill the default value with the merged frequency
    f_orb_f = np.repeat(merge_f, len(f_orb_i))

    # calculate the inner part of the final frequency equation
    inner_part = f_orb_i**(-8/3) - 2**(32/3) * np.pi**(8/3) \
        * t_evol / (5 * c.c**5) * (c.G * m_c)**(5/3) * utils.peters_f(ecc_i)

    # any merged binaries will have a negative inner part
    inspiral = inner_part >= 0.0

    # fill in the values for binaries that are still inspiraling
    f_orb_f[inspiral] = np.power(inner_part[inspiral], -3/8)
    return f_orb_f
