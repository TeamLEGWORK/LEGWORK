"""Functions to calculate signal-to-noise ratio in four different cases"""

import numpy as np
from legwork import strain, psd, utils, evol
import astropy.units as u

__all__ = ['snr_circ_stationary', 'snr_ecc_stationary',
           'snr_circ_evolving', 'snr_ecc_evolving']


def snr_circ_stationary(m_c, f_orb, dist, t_obs, position=None, polarisation=None, inclination=None,
                        interpolated_g=None, interpolated_sc=None, instrument="LISA", custom_psd=None):
    """Computes SNR for circular and stationary sources

    Parameters
    ----------
    m_c : `float/array`
        Chirp mass

    f_orb : `float/array`
        Orbital frequency

    dist : `float/array`
        Distance to the source

    t_obs : `float`
        Total duration of the observation

    position : `SkyCoord/array`, optional
        Sky position of source. Must be specified using Astropy's :class:`astropy.coordinates.SkyCoord` class.

    polarisation : `float/array`, optional
        GW polarisation angle of the source. Must have astropy angular units.

    inclination : `float/array`, optional
        Inclination of the source. Must have astropy angular units.

    interpolated_g : `function`
        A function returned by :class:`scipy.interpolate.interp2d` that computes g(n,e) from Peters (1964).
        The code assumes that the function returns the output sorted as with the interp2d returned functions
        (and thus unsorts). Default is None and uses exact g(n,e) in this case.

    interpolated_sc : `function`
        A function returned by :class:`scipy.interpolate.interp1d` that computes the LISA sensitivity curve.
        Default is None and uses exact values. Note: take care to ensure that your interpolated function has
        the same LISA observation time as ``t_obs`` and uses the same instrument.

    instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
        Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied.

    custom_psd : `function`
        Custom function for computing the PSD. Must take the same arguments as :meth:`legwork.psd.lisa_psd`
        even if it ignores some.

    Returns
    -------
    snr : `float/array`
        SNR for each binary
    """

    # only need to compute n=2 harmonic for circular
    h_0_circ_2 = strain.h_0_n(m_c=m_c, f_orb=f_orb, ecc=np.zeros_like(f_orb).value, n=2, dist=dist,
                              position=position, polarisation=polarisation, inclination=inclination,
                              interpolated_g=interpolated_g).flatten()**2

    h_f_src_circ_2 = h_0_circ_2 * t_obs
    if interpolated_sc is not None:
        h_f_lisa_2 = interpolated_sc(2 * f_orb)
    else:
        h_f_lisa_2 = psd.power_spectral_density(f=2 * f_orb, t_obs=t_obs, instrument=instrument,
                                                custom_psd=custom_psd)
    snr = (h_f_src_circ_2 / h_f_lisa_2)**0.5

    return snr.decompose()


def snr_ecc_stationary(m_c, f_orb, ecc, dist, t_obs, harmonics_required,
                       position=None, polarisation=None, inclination=None,
                       interpolated_g=None, interpolated_sc=None,
                       ret_max_snr_harmonic=False, ret_snr2_by_harmonic=False,
                       instrument="LISA", custom_psd=None):
    """Computes SNR for eccentric and stationary sources

    Parameters
    ----------
    m_c : `float/array`
        Chirp mass

    f_orb : `float/array`
        Orbital frequency

    ecc : `float/array`
        Eccentricity

    dist : `float/array`
        Distance to the source

    t_obs : `float`
        Total duration of the observation

    harmonics_required : `integer`
        Maximum integer harmonic to compute

    position : `SkyCoord/array`, optional
        Sky position of source. Must be specified using Astropy's :class:`astropy.coordinates.SkyCoord` class.

    polarisation : `float/array`, optional
        GW polarisation angle of the source. Must have astropy angular units.

    inclination : `float/array`, optional
        Inclination of the source. Must have astropy angular units.

    interpolated_g : `function`
        A function returned by :class:`scipy.interpolate.interp2d` that computes g(n,e) from Peters (1964).
        The code assumes that the function returns the output sorted as with the interp2d returned functions
        (and thus unsorts). Default is None and uses exact g(n,e) in this case.

    interpolated_sc : `function`
        A function returned by :class:`scipy.interpolate.interp1d` that computes the LISA sensitivity curve.
        Default is None and uses exact values. Note: take care to ensure that your interpolated function has
        the same LISA observation time as ``t_obs`` and uses the same instrument.

    ret_max_snr_harmonic : `boolean`
        Whether to return (in addition to the snr), the harmonic with the maximum SNR

    ret_snr2_by_harmonic : `boolean`
        Whether to return the SNR^2 in each individual harmonic rather than the total.
        The total can be retrieving by summing and then taking the square root.

    instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
        Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied.

    custom_psd : `function`
        Custom function for computing the PSD. Must take the same arguments as :meth:`legwork.psd.lisa_psd`
        even if it ignores some.

    Returns
    -------
    snr : `float/array`
        SNR for each binary

    max_snr_harmonic : `int/array`
        harmonic with maximum SNR for each binary (only returned if ``ret_max_snr_harmonic=True``)
    """
    # define range of harmonics
    n_range = np.arange(1, harmonics_required + 1).astype(int)

    # calculate source signal
    h_0_ecc_n_2 = strain.h_0_n(m_c=m_c, f_orb=f_orb, ecc=ecc, n=n_range, dist=dist,
                               position=position, polarisation=polarisation,
                               inclination=inclination, interpolated_g=interpolated_g)**2

    # reshape the output since only one timestep
    h_0_ecc_n_2 = h_0_ecc_n_2.reshape(len(m_c), harmonics_required)
    h_f_src_ecc_2 = h_0_ecc_n_2 * t_obs

    # calculate harmonic frequencies and noise
    f_n = n_range[np.newaxis, :] * f_orb[:, np.newaxis]
    if interpolated_sc is not None:
        h_f_lisa_n_2 = interpolated_sc(f_n.flatten())
        h_f_lisa_n_2 = h_f_lisa_n_2.reshape(f_n.shape)
    else:
        h_f_lisa_n_2 = psd.power_spectral_density(f=f_n, t_obs=t_obs,
                                                  instrument=instrument, custom_psd=custom_psd)

    snr_n_2 = (h_f_src_ecc_2 / h_f_lisa_n_2).decompose()

    if ret_snr2_by_harmonic:
        return snr_n_2

    # calculate the signal-to-noise ratio
    snr = (np.sum(snr_n_2, axis=1))**0.5

    if ret_max_snr_harmonic:
        max_snr_harmonic = np.argmax(snr_n_2, axis=1) + 1
        return snr, max_snr_harmonic
    else:
        return snr


def snr_circ_evolving(m_1, m_2, f_orb_i, dist, t_obs, n_step,
                      position=None, polarisation=None, inclination=None, t_merge=None,
                      interpolated_g=None, interpolated_sc=None,
                      instrument="LISA", custom_psd=None):
    """Computes SNR for circular and stationary sources

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    f_orb_i : `float/array`
        Initial orbital frequency

    dist : `float/array`
        Distance to the source

    t_obs : `float`
        Total duration of the observation

    n_step : `int`
        Number of time steps during observation duration

    position : `SkyCoord/array`, optional
        Sky position of source. Must be specified using Astropy's :class:`astropy.coordinates.SkyCoord` class.

    polarisation : `float/array`, optional
        GW polarisation angle of the source. Must have astropy angular units.

    inclination : `float/array`, optional
        Inclination of the source. Must have astropy angular units.

    t_merge : `float/array`
        Time until merger

    interpolated_g : `function`
        A function returned by :class:`scipy.interpolate.interp2d` that computes g(n,e) from Peters (1964).
        The code assumes that the function returns the output sorted as with the interp2d returned functions
        (and thus unsorts). Default is None and uses exact g(n,e) in this case.

    interpolated_sc : `function`
        A function returned by :class:`scipy.interpolate.interp1d` that computes the LISA sensitivity curve.
        Default is None and uses exact values. Note: take care to ensure that your interpolated function has
        the same LISA observation time as ``t_obs`` and uses the same instrument.

    instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
        Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied.

    custom_psd : `function`
        Custom function for computing the PSD. Must take the same arguments as :meth:`legwork.psd.lisa_psd`
        even if it ignores some.

    Returns
    -------
    sn : `float/array`
        SNR for each binary
    """
    m_c = utils.chirp_mass(m_1=m_1, m_2=m_2)

    # calculate minimum of observation time and merger time
    if t_merge is None:
        t_merge = evol.get_t_merge_circ(m_1=m_1, m_2=m_2, f_orb_i=f_orb_i)
    t_evol = np.minimum(t_merge - (1 * u.s), t_obs)

    # get f_orb evolution
    f_orb_evol = evol.evol_circ(t_evol=t_evol, n_step=n_step, m_1=m_1, m_2=m_2, f_orb_i=f_orb_i)

    maxes = np.where(f_orb_evol == 1e2 * u.Hz, -1 * u.Hz, f_orb_evol).max(axis=1)
    for source in range(len(f_orb_evol)):
        f_orb_evol[source][f_orb_evol[source] == 1e2 * u.Hz] = maxes[source]

    # calculate the characteristic power
    h_c_n_2 = strain.h_c_n(m_c=m_c, f_orb=f_orb_evol, ecc=np.zeros_like(f_orb_evol).value, n=2, dist=dist,
                           interpolated_g=interpolated_g)**2
    h_c_n_2 = h_c_n_2.reshape(len(m_c), n_step)

    # calculate the characteristic noise power
    if interpolated_sc is not None:
        h_f_lisa_2 = interpolated_sc(2 * f_orb_evol.flatten())
        h_f_lisa_2 = h_f_lisa_2.reshape(f_orb_evol.shape)
    else:
        h_f_lisa_2 = psd.power_spectral_density(f=2 * f_orb_evol, t_obs=t_obs,
                                                instrument=instrument, custom_psd=custom_psd)
    h_c_lisa_2 = (2 * f_orb_evol)**2 * h_f_lisa_2

    snr = np.trapz(y=h_c_n_2 / h_c_lisa_2, x=2 * f_orb_evol, axis=1)**0.5

    return snr.decompose()


def snr_ecc_evolving(m_1, m_2, f_orb_i, dist, ecc, harmonics_required, t_obs, n_step,
                     position=None, polarisation=None, inclination=None, t_merge=None,
                     interpolated_g=None, interpolated_sc=None, n_proc=1,
                     ret_max_snr_harmonic=False, ret_snr2_by_harmonic=False,
                     instrument="LISA", custom_psd=None):
    """Computes SNR for eccentric and evolving sources.

    Note that this function will not work for exactly circular (ecc = 0.0)
    binaries.

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    f_orb_i : `float/array`
        Initial orbital frequency

    dist : `float/array`
        Distance to the source

    ecc : `float/array`
        Eccentricity

    harmonics_required : `int`
        Maximum integer harmonic to compute

    t_obs : `float`
        Total duration of the observation

    position : `SkyCoord/array`, optional
        Sky position of source. Must be specified using Astropy's :class:`astropy.coordinates.SkyCoord` class.

    polarisation : `float/array`, optional
        GW polarisation angle of the source. Must have astropy angular units.

    inclination : `float/array`, optional
        Inclination of the source. Must have astropy angular units.

    n_step : `int`
        Number of time steps during observation duration

    t_merge : `float/array`
        Time until merger

    interpolated_g : `function`
        A function returned by :class:`scipy.interpolate.interp2d` that computes g(n,e) from Peters (1964).
        The code assumes that the function returns the output sorted as with the interp2d returned functions
        (and thus unsorts). Default is None and uses exact g(n,e) in this case.

    interpolated_sc : `function`
        A function returned by :class:`scipy.interpolate.interp1d` that computes the LISA sensitivity curve.
        Default is None and uses exact values. Note: take care to ensure that your interpolated function has
        the same LISA observation time as ``t_obs`` and uses the same instrument.

    n_proc : `int`
        Number of processors to split eccentricity evolution over, where
        the default is n_proc=1

    ret_max_snr_harmonic : `boolean`
        Whether to return (in addition to the snr), the harmonic with the maximum SNR

    ret_snr2_by_harmonic : `boolean`
        Whether to return the SNR^2 in each individual harmonic rather than the total.
        The total can be retrieving by summing and then taking the square root.

    instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
        Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied.

    custom_psd : `function`
        Custom function for computing the PSD. Must take the same arguments as :meth:`legwork.psd.lisa_psd`
        even if it ignores some.

    Returns
    -------
    snr : `float/array`
        SNR for each binary

    max_snr_harmonic : `int/array`
        harmonic with maximum SNR for each binary (only returned if
        ``ret_max_snr_harmonic=True``)
    """
    m_c = utils.chirp_mass(m_1=m_1, m_2=m_2)

    # calculate minimum of observation time and merger time
    if t_merge is None:
        t_merge = evol.get_t_merge_ecc(m_1=m_1, m_2=m_2, f_orb_i=f_orb_i, ecc_i=ecc)

    t_before = 0.1 * u.yr

    t_evol = np.minimum(t_merge - t_before, t_obs).to(u.s)
    # get eccentricity and f_orb evolutions
    e_evol, f_orb_evol = evol.evol_ecc(ecc_i=ecc, t_evol=t_evol, n_step=n_step, m_1=m_1, m_2=m_2,
                                       f_orb_i=f_orb_i, n_proc=n_proc, t_before=t_before, t_merge=t_merge)

    maxes = np.where(np.logical_and(e_evol == 0.0, f_orb_evol == 1e2 * u.Hz),
                     -1 * u.Hz, f_orb_evol).max(axis=1)
    for source in range(len(f_orb_evol)):
        f_orb_evol[source][f_orb_evol[source] == 1e2 * u.Hz] = maxes[source]

    # create harmonics list and multiply for nth frequency evolution
    harms = np.arange(1, harmonics_required + 1).astype(int)
    f_n_evol = harms[np.newaxis, np.newaxis, :] * f_orb_evol[..., np.newaxis]

    # calculate the characteristic strain
    h_c_n_2 = strain.h_c_n(m_c=m_c, f_orb=f_orb_evol, ecc=e_evol, n=harms, dist=dist,
                           position=position, polarisation=polarisation, inclination=inclination,
                           interpolated_g=interpolated_g)**2

    # calculate the characteristic noise power
    if interpolated_sc is not None:
        h_f_lisa = interpolated_sc(f_n_evol.flatten())
    else:
        h_f_lisa = psd.power_spectral_density(f=f_n_evol.flatten(), t_obs=t_obs,
                                              instrument=instrument, custom_psd=custom_psd)
    h_f_lisa = h_f_lisa.reshape(f_n_evol.shape)
    h_c_lisa_2 = f_n_evol**2 * h_f_lisa

    snr_evol = h_c_n_2 / h_c_lisa_2

    # integrate, sum and square root to get SNR
    snr_n_2 = np.trapz(y=snr_evol, x=f_n_evol, axis=1)

    if ret_snr2_by_harmonic:
        return snr_n_2

    snr_2 = snr_n_2.sum(axis=1)
    snr = np.sqrt(snr_2)

    if ret_max_snr_harmonic:
        max_snr_harmonic = np.argmax(snr_n_2, axis=1) + 1
        return snr, max_snr_harmonic
    else:
        return snr
