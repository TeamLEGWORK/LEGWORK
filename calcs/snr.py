"""`snr calcs` for gw calcs"""

import numpy as np
import astropy.units as u
import calcs.strain as strain
import calcs.lisa as lisa
import calcs.utils as utils
import calcs.evol as evol


def snr_circ_stationary(m_c, f_orb, dist, t_obs, interpolated_g=None):
    """Computes the signal to noise ratio for stationary and
    circular binaries

    Params
    ------
    m_c : `float/array`
        chirp mass

    f_orb : `float/array`
        orbital frequency

    dist : `float/array`
        distance to the source

    t_obs : `float`
        total duration of the observation

    interpolated_g : `function`
        A function returned by scipy.interpolate.interp2d that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    sn : `float/array`
        snr for each binary
    """

    # only need to compute n=2 harmonic for circular
    h_0_circ_2 = strain.h_0_n(m_c=m_c, f_orb=f_orb, 
                              ecc=0.0, n=2, dist=dist,
                              interpolated_g=interpolated_g).flatten()**2

    h_f_src_circ_2 = h_0_circ_2 * t_obs
    h_f_lisa_2 = lisa.power_spectral_density(f=2 * f_orb, t_obs=t_obs)
    snr = (h_f_src_circ_2 / (4 * h_f_lisa_2))**0.5

    return snr.decompose()


def snr_ecc_stationary(m_c, f_orb, ecc, dist, t_obs, max_harmonic, interpolated_g=None):
    """Computes the signal to noise ratio for stationary and
    eccentric binaries


    Params
    ------
    m_c : `float/array`
        chirp mass

    f_orb : `float/array`
        orbital frequency

    ecc : `float/array`
        eccentricity

    dist : `float/array`
        distance to the source

    t_obs : `float`
        total duration of the observation

    max_harmonic : `integer`
        maximum integer harmonic to compute

    interpolated_g : `function`
        A function returned by scipy.interpolate.interp2d that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    sn : `float/array`
        sn for each binary
    """
    # define range of harmonics
    n_range = np.arange(1, max_harmonic + 1).astype(int)

    # calculate source signal
    h_0_ecc_n_2 = strain.h_0_n(m_c=m_c, f_orb=f_orb,
                               ecc=ecc, n=n_range, dist=dist,
                               interpolated_g=interpolated_g)**2
    h_f_src_ecc_2 = h_0_ecc_n_2 * t_obs

    # turn n_range into grid and calcualte noise
    N, F = np.meshgrid(n_range, f_orb)
    h_f_lisa_n_2 = lisa.power_spectral_density(f=N * F, t_obs=t_obs)

    # calculate the signal-to-noise ratio
    snr = (np.sum(h_f_src_ecc_2 / (4*h_f_lisa_n_2), axis=1))**0.5
    return snr.decompose()


def snr_circ_evolving(m_1, m_2, f_orb_i, dist, t_obs, n_step, interpolated_g=None):
    """Computes the signal to noise ratio for stationary and
    circular binaries

    Params
    ------
    m_1 : `float/array`
        primary mass

    m_2 : `float/array`
        secondary mass

    f_orb_i : `float/array`
        initial orbital frequency

    dist : `float/array`
        distance to the source

    t_obs : `float`
        total duration of the observation

    n_step : `int`
        number of timesteps during obsrvation duration

    interpolated_g : `function`
        A function returned by scipy.interpolate.interp2d that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    sn : `float/array`
        snr for each binary
    """

    m_c = utils.chirp_mass(m_1=m_1, m_2=m_2)

    # calculate minimum of observation time and merger time
    t_merge = evol.get_t_merge_circ(m_1=m_1,
                                    m_2=m_2,
                                    f_orb_i=f_orb_i)
    t_evol = np.where(t_merge < t_obs, t_merge, t_obs)

    # get f_orb, ecc evolution
    f_evol, e_evol = evol.get_f_and_e(m_1=m_1,
                                      m_2=m_2,
                                      f_orb_i=f_orb_i,
                                      e_i=0,
                                      t_evol=t_evol,
                                      circ_tol=0.1,
                                      n_step=n_step)

    # calculate the characteristic power
    h_c_n_2 = strain.h_c_n(m_c=m_c,
                           f_orb=f_evol,
                           ecc=np.zeros(len(m_c)),
                           n=2,
                           dist=dist,
                           interpolated_g=interpolated_g)**2
    # calculate the characteristic noise power
    h_f_lisa_2 = lisa.power_spectral_density(f=2 * f_evol, t_obs=t_obs)
    h_c_lisa_2 = 4 * (2*f_evol)**2 * h_f_lisa_2

    snr = (np.sum(h_c_n_2[:-1] / h_c_lisa_2[:-1] * (f_evol[1:] - f_evol[:-1]), axis=0))**(0.5)

    return snr.decompose()

def snr_ecc_evolving(m_1, m_2, f_orb_i, dist, ecc, max_harmonic, t_obs, n_step, interpolated_g=None):
    """Computes the signal to noise ratio for stationary and
    eccentric binaries


    Params
    ------
    m_1 : `float/array`
        primary mass

    m_2 : `float/array`
        secondary mass

    f_orb_i : `float/array`
        initial orbital frequency

    dist : `float/array`
        distance to the source

    ecc : `float/array`
        eccentricity

    max_harmonic : `int`
        maximum integer harmonic to compute

    t_obs : `float`
        total duration of the observation

    n_step : `int`
        number of timesteps during obsrvation duration

    interpolated_g : `function`
        A function returned by scipy.interpolate.interp2d that
        computes g(n,e) from Peters (1964). The code assumes
        that the function returns the output sorted as with the
        interp2d returned functions (and thus unsorts).
        Default is None and uses exact g(n,e) in this case.

    Returns
    -------
    sn : `array`
        snr for each binary
    """
    raise NotImplementedError("Katie todo")