"""Functions using equations from Peters and Mathews (1964) to calculate inspiral times and evolve
binary parameters."""

import legwork.utils as utils
from numba import jit, njit
import numba as nb
from scipy.integrate import odeint, quad
import numpy as np
import astropy.units as u
import astropy.constants as c
from schwimmbad import MultiPool

__all__ = ['de_dt', 'integrate_de_dt', 'evol_circ', 'evol_ecc', 'get_t_merge_circ', 'get_t_merge_ecc',
           't_merge_mandel_fit', 'evolve_f_orb_circ', 'check_mass_freq_input', 'create_timesteps_array',
           'determine_stationarity']


@jit
def de_dt(e, times, beta, c_0):                             # pragma: no cover
    """Compute eccentricity time derivative

    Computes the evolution of the eccentricity from the emission of gravitational waves following
    Peters & Mathews (1964) Eq. 5.13

    Parameters
    ----------
    e : `float`
        Initial eccentricity

    times : `float/array`
        Evolution timestep. Not actually used in function but required for use with scipy's
        :func:`scipy.integrate.odeint`

    beta : `float`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`

    c_0 : `float`
        Constant defined in Peters and Mathews (1964) Eq. 5.11. See :meth:`legwork.utils.c_0`

    Returns
    -------
    dedt : `float/array`
        Eccentricity time derivative
    """
    dedt = -19 / 12 * beta / c_0**4 * (e**(-29 / 19) * (1 - e**2)**(3/2)) \
        / (1 + (121/304) * e**2)**(1181/2299)
    return dedt


def integrate_de_dt(args):                         # pragma: no cover
    """Wrapper that integrates :func:`legwork.evol.de_dt` with odeint

    Parameters
    ----------
    args : `list`
        List of arguments for :func:`legwork.evol.de_dt` including [e, times, beta, c_0]

    Returns
    -------
    ecc_evol : `array`
       eccentricity evolution
    """
    ecc_i, timesteps, beta, c_0 = args
    ecc_evol = odeint(de_dt, ecc_i, timesteps, args=(beta, c_0)).flatten()
    return ecc_evol


def check_mass_freq_input(beta=None, m_1=None, m_2=None,
                          a_i=None, f_orb_i=None):
    """Check that mass and frequency input is valid

    Helper function to check that either ``beta`` or (``m_1`` and ``m_2``) is provided and that ``a_i`` or
    ``f_orb_i`` is provided as well as calculate quantities that are not passed as arguments.

    Parameters
    ----------

    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta` (if supplied
        ``m_1`` and `m_2` are ignored)

    m_1 : `float/array`
        Primary mass (required if ``beta`` is None)

    m_2 : `float/array`
        Secondary mass (required if ``beta`` is None)

    a_i : `float/array`
        Initial semi-major axis (if supplied ``f_orb_i`` is ignored)

    f_orb_i : `float/array`
        Initial orbital frequency (required if ``a_i`` is None)

    Returns
    -------
    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`

    a_i : `float/array`
        Initial semi-major axis
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
        raise ValueError("Individual masses `m_1` and `m_2` are required if no value of `a_i` is supplied")
    elif a_i is None:
        a_i = utils.get_a_from_f_orb(f_orb=f_orb_i, m_1=m_1, m_2=m_2)
    return beta, a_i


def create_timesteps_array(a_i, beta, ecc_i=None, t_evol=None, n_step=100, timesteps=None):
    """Create an array of timesteps

    Parameters
    ----------
    a_i : `float/array`
        Initial semi-major axis

    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`

    ecc_i : `float/array`
        Initial eccentricity

    t_evol : `float/array`
        Amount of time for which to evolve each binaries. Required if ``timesteps`` is None. If None,
        then defaults to merger times.

    n_steps : `int`
        Number of timesteps to take between t=0 and t=``t_evol``. Required if ``timesteps`` is None.
        Defaults to 100.

    timesteps : `float/array`
        Array of exact timesteps to take when evolving each binary. Must be monotonically increasing and
        start with t=0. Either supply a 1D array to use for every binary or a 2D array that has a different
        array of timesteps for each binary. ``timesteps`` is used in place of ``t_evol`` and ``n_steps``
        and takes precedence over them.

    Returns
    -------
    timesteps : `float/array`
        Array of timesteps for each binary
    """
    # create timesteps array if not provided
    if timesteps is None:
        # if no evolution times given, use merger times
        if t_evol is None:
            t_evol = get_t_merge_ecc(ecc_i=ecc_i, a_i=a_i, beta=beta)
        # if only one time, repeat for every binary
        elif not isinstance(t_evol.value, np.ndarray):
            t_evol = np.repeat(t_evol.value, len(a_i)) * t_evol.unit
        timesteps = np.linspace(0 * u.s, t_evol, n_step).T
    # broadcast the times to every source if only one array provided
    elif np.ndim(timesteps) == 1:
        timesteps = timesteps[np.newaxis, :]
        if isinstance(a_i.value, np.ndarray):
            timesteps = np.broadcast_to(timesteps.value, (len(a_i), len(timesteps[0]))) * timesteps.unit
    return timesteps


def evol_circ(t_evol=None, n_step=100, timesteps=None, beta=None, m_1=None, m_2=None,
              a_i=None, f_orb_i=None, output_vars='f_orb'):
    """Evolve an array of circular binaries for ``t_evol`` time

    This function implements Peters & Mathews (1964) Eq. 5.9.

    Note that all of {``beta``, ``m_1``, ``m_2``, ``a_i``, ``f_orb_i``} must have the same dimensions.

    Parameters
    ----------

    t_evol : `float/array`
        Amount of time for which to evolve each binaries. Required if ``timesteps`` is None. Defaults to
        merger times.

    n_steps : `int`
        Number of timesteps to take between t=0 and t=``t_evol``. Required if ``timesteps`` is None.
        Defaults to 100.

    timesteps : `float/array`
        Array of exact timesteps to take when evolving each binary. Must be monotonically increasing and
        start with t=0. Either supply a 1D array to use for every binary or a 2D array that has a different
        array of timesteps for each binary. ``timesteps`` is used in place of ``t_evol`` and ``n_steps``
        and takes precedence over them.

    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`
        (if supplied ``m_1`` and `m_2` are ignored)

    m_1 : `float/array`
        Primary mass (required if ``beta`` is None or if ``output_vars`` contains a frequency)

    m_2 : `float/array`
        Secondary mass (required if ``beta`` is None or if ``output_vars`` contains a frequency)

    a_i : `float/array`
        Initial semi-major axis (if supplied ``f_orb_i`` is ignored)

    f_orb_i : `float/array`
        Initial orbital frequency (required if ``a_i`` is None)

    output_vars : `str/array`
        List of **ordered** output vars, or a single var. Choose from any of ``timesteps``, ``a``, ``f_orb``
        and ``f_GW`` for which of timesteps, semi-major axis and orbital/GW frequency that you want.
        Default is ``f_orb``.

    Returns
    -------
    evolution : `array`
        Array containing any of semi-major axis, timesteps and frequency evolution. Content determined by
        ``output_vars``.
    """
    # transform input if only a single source
    arrayed_args, single_source = utils.ensure_array(m_1, m_2, beta, a_i, f_orb_i)
    m_1, m_2, beta, a_i, f_orb_i = arrayed_args
    output_vars = np.array([output_vars]) if isinstance(output_vars, str) else output_vars

    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2, a_i=a_i, f_orb_i=f_orb_i)

    if np.isin(output_vars, ["f_orb", "f_GW"]).any() and (m_1 is None or m_2 is None):
        raise ValueError("`m_1`` and `m_2` required if `output_vars` contains a frequency")
    timesteps = create_timesteps_array(a_i=a_i, beta=beta, ecc_i=np.zeros_like(a_i), t_evol=t_evol,
                                       n_step=n_step, timesteps=timesteps)

    # perform the evolution
    difference = a_i[:, np.newaxis]**4 - 4 * beta[:, np.newaxis] * timesteps
    difference = np.where(difference.value <= 0.0, 0.0, difference)
    a_evol = difference**(1/4)

    # calculate f_orb_evol if any frequency requested
    if np.isin(output_vars, ["f_orb", "f_GW"]).any():
        # change merged binaries to extremely small separations
        a_not0 = np.where(a_evol.value == 0.0, 1e-30 * a_evol.unit, a_evol)
        f_orb_evol = utils.get_f_orb_from_a(a=a_not0, m_1=m_1[:, np.newaxis], m_2=m_2[:, np.newaxis])

        # change frequencies back to 1Hz since LISA can't measure above
        f_orb_evol = np.where(a_not0.value == 1e-30, 1e2 * u.Hz, f_orb_evol)

    # construct evolution output
    evolution = []
    for var in output_vars:
        if var == "timesteps":
            timesteps = timesteps.flatten() if single_source else timesteps
            evolution.append(timesteps.to(u.yr))
        elif var == "a":
            a_evol = a_evol.flatten() if single_source else a_evol
            evolution.append(a_evol.to(u.AU))
        elif var == "f_orb":
            f_orb_evol = f_orb_evol.flatten() if single_source else f_orb_evol
            evolution.append(f_orb_evol.to(u.Hz))
        elif var == "f_GW":
            f_orb_evol = f_orb_evol.flatten() if single_source else f_orb_evol
            evolution.append(2 * f_orb_evol.to(u.Hz))
    return evolution if len(evolution) > 1 else evolution[0]


def evol_ecc(ecc_i, t_evol=None, n_step=100, timesteps=None, beta=None, m_1=None, m_2=None,
             a_i=None, f_orb_i=None, output_vars=['ecc', 'f_orb'], n_proc=1,
             avoid_merger=True, exact_t_merge=False, t_before=1 * u.Myr, t_merge=None):
    """Evolve an array of eccentric binaries for ``t_evol`` time

    This function use Peters & Mathews (1964) Eq. 5.11 and 5.13.

    Note that all of {``beta``, ``m_1``, ``m_2``, ``ecc_i``, ``a_i``, ``f_orb_i``} must have the
    same dimensions.

    Parameters
    ----------
    ecc_i : `float/array`
        Initial eccentricity

    t_evol : `float/array`
        Amount of time for which to evolve each binaries. Required if ``timesteps`` is None. Defaults to
        merger times.

    n_steps : `int`
        Number of timesteps to take between t=0 and t=``t_evol``. Required if ``timesteps`` is None.
        Defaults to 100.

    timesteps : `float/array`
        Array of exact timesteps to take when evolving each binary. Must be monotonically increasing and
        start with t=0. Either supply a 1D array to use for every binary or a 2D array that has a different
        array of timesteps for each binary. ``timesteps`` is used in place of ``t_evol`` and ``n_steps`` and
        takes precedence over them.

    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`
        (if supplied ``m_1`` and `m_2` are ignored)

    m_1 : `float/array`
        Primary mass (required if ``beta`` is None or if ``output_vars`` contains a frequency)

    m_2 : `float/array`
        Secondary mass (required if ``beta`` is None or if ``output_vars`` contains a frequency)

    a_i : `float/array`
        Initial semi-major axis (if supplied ``f_orb_i`` is ignored)

    f_orb_i : `float/array`
        Initial orbital frequency (required if ``a_i`` is None)

    output_vars : `array`
        List of **ordered** output vars, choose from any of ``timesteps``, ``ecc``, ``a``, ``f_orb`` and
        ``f_GW`` for which of timesteps, eccentricity, semi-major axis and orbital/GW frequency that you want.
        Default is [``ecc``, ``f_orb``]

    n_proc : `int`
        Number of processors to split eccentricity evolution over, where the default is n_proc=1

    avoid_merger : `boolean`
        Whether to avoid integration around the merger of the binary. Warning:
        setting this to false will result in many LSODA errors to be outputted
        since the derivatives get so large.

    exact_t_merge : `boolean`
        Whether to calculate the merger time exactly or use a fit (only
        relevant when ``avoid_merger`` is set to True

    t_before : `float`
        How much time before the merger to cutoff the integration (default is
        1 Myr - this will prevent all LSODA warnings for e < 0.95, you may
        need to increase this time if your sample is more eccentric than this)

    t_merge : `float/array`
        Merger times for each source to be evolved. Only used when
        `avoid_merger=True`. If `None` then these will be automatically
        calculated either approximately or exactly based on the values of
        `exact_t_merge`.

    Returns
    -------
    evolution : `array`
        Array possibly containing eccentricity, semi-major axis, timesteps and frequency evolution.
        Content determined by ``output_vars``
    """
    # transform input if only a single source
    arrayed_args, single_source = utils.ensure_array(m_1, m_2, beta, a_i, f_orb_i, ecc_i)
    m_1, m_2, beta, a_i, f_orb_i, ecc_i = arrayed_args
    output_vars = np.array([output_vars]) if isinstance(output_vars, str) else output_vars

    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2, a_i=a_i, f_orb_i=f_orb_i)

    if np.isin(output_vars, ["f_orb", "f_GW"]).any() and (m_1 is None or m_2 is None):
        raise ValueError("`m_1`` and `m_2` required if `output_vars` contains a frequency")

    c_0 = utils.c_0(a_i=a_i, ecc_i=ecc_i)
    timesteps = create_timesteps_array(a_i=a_i, beta=beta, ecc_i=ecc_i,
                                       t_evol=t_evol, n_step=n_step, timesteps=timesteps)

    # if avoiding the merger during integration
    if avoid_merger:
        if t_merge is None:
            # calculate the merger time
            t_merge = get_t_merge_ecc(ecc_i=ecc_i, a_i=a_i,
                                      beta=beta, exact=exact_t_merge).to(u.Gyr)

        # make a mask for any timesteps that are too close to the merger
        too_close = timesteps >= t_merge[:, np.newaxis] - t_before

        check = too_close
        check[:, 0] = True
        if np.all(check):           # pragma: no cover
            print("WARNING: All timesteps are too close to merger so",
                  "evolution is not possible. Either set `t_before` to a",
                  "smaller time or turn off `avoid_merger`")

        # ensure that the first timestep is always valid
        too_close[:, 0] = False

        if np.any(too_close):
            # set them all equal to the previous timestep before passing limit
            timesteps[too_close] = -1 * u.Gyr
            previous = timesteps.max(axis=1).repeat(timesteps.shape[1])
            timesteps[too_close] = previous.reshape(timesteps.shape)[too_close]

    # get rid of the units for faster integration
    c_0 = c_0.to(u.m).value
    beta = beta.to(u.m**4 / u.s).value
    timesteps = timesteps.to(u.s).value

    # perform the evolution
    if n_proc > 1:
        with MultiPool(processes=n_proc) as pool:
            ecc_evol = np.array(list(pool.map(integrate_de_dt, zip(ecc_i, timesteps.tolist(), beta, c_0))))
    else:
        ecc_evol = np.array([odeint(de_dt, ecc_i[i], timesteps[i], args=(beta[i], c_0[i])).flatten()
                             for i in range(len(ecc_i))])

    c_0 = c_0[:, np.newaxis] * u.m
    ecc_evol = np.nan_to_num(ecc_evol, nan=0.0)

    # calculate a_evol if any frequency or separation requested
    if np.isin(output_vars, ["a", "f_orb", "f_GW"]).any():
        a_evol = utils.get_a_from_ecc(ecc_evol, c_0)

        # calculate f_orb_evol if any frequency requested
        if np.isin(output_vars, ["f_orb", "f_GW"]).any():
            # change merged binaries to extremely small separations
            a_not0 = np.where(a_evol.value == 0.0, 1e-30 * a_evol.unit, a_evol)
            f_orb_evol = utils.get_f_orb_from_a(a=a_not0, m_1=m_1[:, np.newaxis], m_2=m_2[:, np.newaxis])

            # change frequencies back to 1Hz since LISA can't measure above
            f_orb_evol = np.where(a_not0.value == 1e-30, 1e2 * u.Hz, f_orb_evol)

    # construct evolution output
    evolution = []
    for var in output_vars:
        if var == "timesteps":
            timesteps = timesteps.flatten() if single_source else timesteps
            evolution.append((timesteps * u.s).to(u.yr))
        elif var == "ecc":
            ecc_evol = ecc_evol.flatten() if single_source else ecc_evol
            evolution.append(ecc_evol)
        elif var == "a":
            a_evol = a_evol.flatten() if single_source else a_evol
            evolution.append(a_evol.to(u.AU))
        elif var == "f_orb":
            f_orb_evol = f_orb_evol.flatten() if single_source else f_orb_evol
            evolution.append(f_orb_evol.to(u.Hz))
        elif var == "f_GW":
            f_orb_evol = f_orb_evol.flatten() if single_source else f_orb_evol
            evolution.append(2 * f_orb_evol.to(u.Hz))
    return evolution if len(evolution) > 1 else evolution[0]


def get_t_merge_circ(beta=None, m_1=None, m_2=None,
                     a_i=None, f_orb_i=None):
    """Computes the merger time for circular binaries

    This function implements Peters & Mathews (1964) Eq. 5.10

    Parameters
    ----------
    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`
        (if supplied `m_1` and `m_2` are ignored)

    m_1 : `float/array`
        Primary mass (required if `beta` is None)

    m_2 : `float/array`
        Secondary mass (required if `beta` is None)

    a_i : `float/array`
        Initial semi-major axis (if supplied `f_orb_i` is ignored)

    f_orb_i : `float/array`
        Initial orbital frequency (required if `a_i` is None)

    Returns
    -------
    t_merge : `float/array`
        Merger time
    """
    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2, a_i=a_i, f_orb_i=f_orb_i)

    # apply Peters 1964 Eq. 5.9
    t_merge = a_i**4 / (4 * beta)

    return t_merge.to(u.Gyr)


def t_merge_mandel_fit(ecc_i):
    """A fit to the Peters 1964 merger time equation (5.14) by Ilya Mandel. This function gives a factor
    which, when multiplied by the circular merger time, gives the eccentric merger time with 3% errors.
    We add a rudimentary polynomial fit to further reduce these errors to within 0.5%.
    ADS Link: https://ui.adsabs.harvard.edu/abs/2021RNAAS...5..223M/abstract

    Parameters
    ----------
    ecc_i : `float/array`
        Initial eccentricity

    Returns
    -------
    factor : `float/array`
        Factor by which to multiply the circular merger timescale by to get the overall merger time.
    """
    coefficients = np.array([-1.20317749e+03, 5.67211219e+03, -1.13935479e+04,
                             1.27306422e+04, -8.66281737e+03,  3.69447796e+03,
                             -9.79864734e+02,  1.54873214e+02, -1.32267683e+01,
                             5.32494018e-01,  9.93382093e-01])
    n_coeff = len(coefficients)
    correction = np.array([coefficients[i] * np.power(ecc_i, n_coeff - (i + 1))
                           for i in range(n_coeff)]).sum(axis=0)

    return (1 + (0.27 * ecc_i**10)
            + (0.33 * ecc_i**20)
            + (0.20 * ecc_i**1000)) * (1 - ecc_i**2)**(7/2) / correction


def get_t_merge_ecc(ecc_i, a_i=None, f_orb_i=None, beta=None, m_1=None, m_2=None,
                    small_e_tol=0.15, large_e_tol=1 - 1e-4, exact=True):
    """Computes the merger time for binaries

    This function implements Peters (1964) Eq. 5.10, 5.14 and the two unlabelled equations after 5.14
    (using a different one depending on the eccentricity of each binary)

    Parameters
    ----------
    ecc_i : `float/array`
        Initial eccentricity (if `ecc_i` is known to be 0.0 then use `get_t_merge_circ` instead)

    a_i : `float/array`
        Initial semi-major axis (if supplied `f_orb_i` is ignored)

    f_orb_i : `float/array`
        Initial orbital frequency (required if `a_i` is None)

    beta : `float/array`
        Constant defined in Peters and Mathews (1964) Eq. 5.9. See :meth:`legwork.utils.beta`
        (if supplied `m_1` and `m_2` are ignored)

    m_1 : `float/array`
        Primary mass (required if `beta` is None)

    m_2 : `float/array`
        Secondary mass (required if `beta` is None)

    small_e_tol : `float`
        Eccentricity below which to apply the small e approximation (see first unlabelled equation following
        Eq. 5.14 of Peters 1964), defaults to 0.15 to keep relative error below approximately 2%

    large_e_tol : `float`
        Eccentricity above which to apply the large e approximation (see second unlabelled equation following
        Eq. 5.14 of Peters 1964), defaults to 0.9999 to keep relative error below approximately 2%

    exact : `boolean`
        Whether to calculate the merger time exactly with numerical integration or to instead use the fit from
        Mandel 2021 (see :meth:`legwork.evol.t_merge_mandel_fit`)

    Returns
    -------
    t_merge : `float/array`
        Merger time
    """
    beta, a_i = check_mass_freq_input(beta=beta, m_1=m_1, m_2=m_2, a_i=a_i, f_orb_i=f_orb_i)

    # shortcut if all binaries are circular
    if np.all(ecc_i == 0.0):
        return get_t_merge_circ(beta=beta, a_i=a_i)

    @njit
    def peters_5_14(e):                                 # pragma: no cover
        """ merger time from Peters Eq. 5.14 """
        return np.power(e, 29/19) * np.power(1 + (121/304)*e**2, 1181/2299) / np.power(1 - e**2, 3/2)

    # case with array of binaries
    if isinstance(ecc_i, (np.ndarray, list)):
        # mask eccentricity based on tolerances
        circular = ecc_i == 0.0
        small_e = np.logical_and(ecc_i > 0.0, ecc_i < small_e_tol)
        large_e = ecc_i > large_e_tol
        other_e = np.logical_and(ecc_i >= small_e_tol, ecc_i <= large_e_tol)

        # calculate c0 from Peters Eq. 5.11 (avoid circular binaries)
        c0 = np.zeros_like(a_i)
        not_circ = np.logical_not(circular)
        c0[not_circ] = utils.c_0(a_i[not_circ], ecc_i[not_circ])

        t_merge = np.zeros(len(ecc_i)) * u.Gyr

        # merger time for circular binaries (Peters Eq. 5.9)
        t_merge[circular] = a_i[circular]**4 / (4 * beta[circular])

        # merger time for low e binaries (Eq after Peters Eq. 5.14)
        t_merge[small_e] = c0[small_e]**4 / (4 * beta[small_e]) * ecc_i[small_e]**(48/19)

        # merger time for high e binaries (2nd Eq after Peters Eq. 5.14)
        t_merge[large_e] = c0[large_e]**4 / (4 * beta[large_e]) \
            * ecc_i[large_e]**(48/19) * (768 / 425) \
            * (1 - ecc_i[large_e]**2)**(-1/2) \
            * (1 + 121/304 * ecc_i[large_e]**2)**(3480/2299)

        # merger time for general binaries (Peters Eq. 5.14)
        if exact:
            prefac = ((12 / 19) * c0[other_e]**4 / beta[other_e]).to(u.Gyr)
            t_merge[other_e] = prefac * [quad(peters_5_14, 0, ecc_i[other_e][i])[0]
                                         for i in range(len(ecc_i[other_e]))]
        else:
            t_merge[other_e] = get_t_merge_circ(beta=beta[other_e],
                                                a_i=a_i[other_e]) \
                * t_merge_mandel_fit(ecc_i[other_e])
    # case with only one binary
    else:
        # calculate c0 from Peters Eq. 5.11
        c0 = utils.c_0(a_i, ecc_i)

        # conditions as above (no need for ecc=0.0 since it never reaches here)
        if ecc_i < small_e_tol:
            t_merge = c0**4 / (4 * beta) * ecc_i**(48/19)
        elif ecc_i > large_e_tol:
            t_merge = c0**4 / (4 * beta) * ecc_i**(48/19) * (768 / 425) \
                * (1 - ecc_i**2)**(-1/2) \
                * (1 + 121/304 * ecc_i**2)**(3480/2299)
        elif exact:
            t_merge = ((12 / 19) * c0**4 / beta * quad(peters_5_14, 0, ecc_i)[0])
        else:
            t_merge = get_t_merge_circ(beta=beta, a_i=a_i) * t_merge_mandel_fit(ecc_i)
    return t_merge.to(u.Gyr)


def evolve_f_orb_circ(f_orb_i, m_c, t_evol, ecc_i=0.0, merge_f=1e9 * u.Hz):
    """Evolve orbital frequency for ``t_evol`` time.

    This gives the exact final frequency for circular binaries. However, it will overestimate the final
    frequency for an eccentric binary and if an exact value is required then :func:`legwork.evol.evol_ecc()`
    should be used instead.

    Parameters
    ----------
    f_orb_i : `float/array`
        Initial orbital frequency

    m_c : `float/array`
        Chirp mass

    t_evol : `float`
        Time over which the frequency evolves

    ecc_i : `float/array`
        Initial eccentricity

    merge_f : `float`
        Frequency to assign if the binary has already merged after ``t_evol``

    Returns
    -------
    f_orb_f : `bool/array`
        Final orbital frequency
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


def determine_stationarity(f_orb_i, t_evol, ecc_i, m_1=None, m_2=None, m_c=None, stat_tol=1e-2):
    """Determine whether a binary is stationary

    Check how much a binary's orbital frequency changes over ``t_evol`` time. This function provides a
    conservative estimate in that some binaries that are stationary may be marked as evolving. This is
    because the eccentricity also evolves but only use the initial value. Solving this in full would
    require the same amount of time as assuming the binary is evolving.

    Parameters
    ----------
    forb_i : `float/array`
        Initial orbital frequency

    t_evol : `float`
        Time over which the frequency evolves

    ecc : `float/array`
        Initial eccentricity

    m_1 : `float/array`
        Primary mass (required if ``m_c`` is None)

    m_2 : `float/array`
        Secondary mass (required if ``m_c`` is None)

    m_c : `float/array`
        Chirp mass (overrides `m_1` and `m_2`)

    stat_tol : `float`
        Fractional change in frequency above which we do not consider a binary to be stationary

    Returns
    -------
    stationary : `bool/array`
        Mask of whether each binary is stationary
    """
    # calculate chirp mass if necessary
    if m_c is None:
        if m_1 is None or m_1 is None:
            raise ValueError("`m_1` and `m_2` are required if `m_c` is None")
        m_c = utils.chirp_mass(m_1, m_2)

    # calculate the final frequency
    f_orb_f = evolve_f_orb_circ(f_orb_i=f_orb_i, m_c=m_c, t_evol=t_evol, ecc_i=ecc_i)

    # check the stationary criterion
    stationary = (f_orb_f - f_orb_i) / f_orb_i <= stat_tol
    return stationary
