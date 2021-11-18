"""Functions to compute various power spectral densities for sensitivity
curves"""

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import splev, splrep
from importlib import resources

__all__ = ['load_response_function', 'approximate_response_function', 'power_spectral_density',
           'lisa_psd', 'tianqin_psd']


def load_response_function(f, fstar=19.09e-3):
    """Load in LISA response function from file

    Load response function and interpolate values for a range of frequencies.
    Adapted from https://github.com/eXtremeGravityInstitute/LISA_Sensitivity
    to use binary files instead of text. See Robson+19 for more details.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    Returns
    -------
    R : `float/array`
        LISA response function at each frequency
    """
    # try to load the values for interpolating R
    try:
        with resources.path(package="legwork", resource="R.npy") as path:
            f_R, R = np.load(path)
    except FileExistsError:  # pragma: no cover
        print("WARNING: Can't find response function file, using approximation instead")
        return approximate_response_function(f, fstar)

    # interpolate the R values in the file
    R_data = splrep(f_R * fstar, R, s=0)

    # use interpolated curve to get R values for supplied f values
    R = splev(f, R_data, der=0)
    return R


def approximate_response_function(f, fstar):
    """Approximate LISA response function

    Use Eq.9 of Robson+19 to approximate the LISA response function.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    Returns
    -------
    R : `float/array`
        response function at each frequency
    """
    return (3 / 10) / (1 + 0.6 * (f / fstar)**2)


def lisa_psd(f, t_obs=4 * u.yr, L=2.5e9 * u.m, approximate_R=False, confusion_noise="robson19"):
    """Calculates the effective LISA power spectral density sensitivity curve

    Using equations from Robson+19, calculate the effective LISA power spectral
    density for the sensitivity curve

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    t_obs : `float`
        Observation time (default 4 years)

    L : `float`
        LISA arm length in metres (default = 2.5Gm)

    approximate_R : `boolean`
        Whether to approximate the response function (default: no)

    confusion_noise  : `various`
        Galactic confusion noise. Acceptable inputs are 'robson19' (the confusion noise from Robson+19),
        `None` (don't include confusion noise) or a custom function that gives the confusion noise at each
        frequency for a given mission length where it would be called by running `noise(f, t_obs)`

    Returns
    -------
    Sn : `float/array`
        Effective power strain spectral density
    """
    # convert frequency from Hz to float for calculations
    f = f.to(u.Hz).value

    # minimum and maximum frequencies in Hz based on the R file from Robson+19
    MIN_F = 1e-7
    MAX_F = 2e0

    # overwrite frequencies that outside the range to prevent error
    f = np.where(np.logical_and(f >= MIN_F, f <= MAX_F), f, 1e-8)

    # single link optical metrology noise (Robson+ Eq. 10)
    def Poms(f):
        return (1.5e-11)**2 * (1 + (2e-3 / f)**4)

    # single test mass acceleration noise (Robson+ Eq. 11)
    def Pacc(f):
        return (3e-15)**2 * (1 + (0.4e-3 / f)**2) * (1 + (f / (8e-3))**4)

    # galactic confusion noise (Robson+ Eq. 14)
    def Sc(f, t_obs):

        # parameters from Robson+ 2019 Table 1
        lengths = np.array([0.5, 1.0, 2.0, 4.0]) * u.yr
        alpha = [0.133, 0.171, 0.165, 0.138]
        beta = [243.0, 292.0, 299.0, -221.0]
        kappa = [482.0, 1020.0, 611.0, 521.0]
        gamma = [917.0, 1680.0, 1340.0, 1680.0]
        fk = [2.58e-3, 2.15e-3, 1.73e-3, 1.13e-3]

        # find index of the closest length to inputted observation time
        ind = np.abs(t_obs - lengths).argmin()

        return 9e-45 * f**(-7 / 3.) * np.exp(-f**(alpha[ind]) + beta[ind] * f * np.sin(kappa[ind] * f)) \
            * (1 + np.tanh(gamma[ind] * (fk[ind] - f)))

    # calculate response function (either exactly or with approximation)
    fstar = (const.c / (2 * np.pi * L)).to(u.Hz).value
    if approximate_R:
        R = approximate_response_function(f, fstar)
    else:
        R = load_response_function(f, fstar)

    # use confusion noise from Robson+19
    if confusion_noise == "robson19":
        cn = Sc(f, t_obs)
    # use a custom function for the confusion noise
    elif confusion_noise is not None:
        cn = confusion_noise(f, t_obs)
    # don't include any confusion noise
    else:
        cn = np.zeros(np.shape(f)) if isinstance(f, (list, np.ndarray)) else 0.0

    L = L.to(u.m).value

    # calculate sensitivity curve
    psd = (1 / (L**2) * (Poms(f) + 4 * Pacc(f) / (2 * np.pi * f)**4)) / R + cn

    # replace values for bad frequencies (set to extremely high value)
    psd = np.where(np.logical_and(f >= MIN_F, f <= MAX_F), psd, np.inf)
    return psd / u.Hz


def tianqin_psd(f, L=np.sqrt(3) * 1e5 * u.km, t_obs=None, approximate_R=None, confusion_noise=None):
    """Calculates the effective TianQin power spectral density sensitivity curve

    Using Eq. 13 from Huang+20, calculate the effective TianQin PSD for the sensitivity curve

    Note that this function includes an extra factor of 10/3 compared Eq. 13 in Huang+20, since Huang+20
    absorbs the factor into the waveform but we instead follow the same convention as Robson+19 for
    consistency and include it in this 'effective' PSD function instead.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    L : `float`
        Arm length

    t_obs : `float`
        Ignored for this function

    approximate_R : `boolean`
        Ignored for this function

    confusion_noise  : `various`
        Ignored for this function

    Returns
    -------
    psd : `float/array`
        Effective power strain spectral density
    """
    fstar = const.c / (2 * np.pi * L)
    Sa = 1e-30 * u.m**2 * u.s**(-4) * u.Hz**(-1)
    Sx = 1e-24 * u.m**2 * u.Hz**(-1)
    psd = 10 / (3 * L**2) * (4 * Sa / (2 * np.pi * f)**4 * (1 + (1e-4 * u.Hz / f)) + Sx) \
        * (1 + 0.6 * (f / fstar)**2)
    return psd.to(u.Hz**(-1))


def power_spectral_density(f, instrument="LISA", custom_psd=None, t_obs=4 * u.yr, L=None,
                           approximate_R=False, confusion_noise="robson19"):
    """Calculates the effective power spectral density for all instruments.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    instrument: {{ `LISA`, `TianQin`, `custom` }}
        Instrument to use. LISA is used by default. Choosing `custom` uses ``custom_psd`` to compute PSD.

    custom_psd : `function`
        Custom function for computing the PSD. Must take the same arguments as :meth:`legwork.psd.lisa_psd`
        even if it ignores some.

    t_obs : `float`
        Observation time (default 4 years)

    L : `float`
        LISA arm length in metres

    approximate_R : `boolean`
        Whether to approximate the response function (default: no)

    confusion_noise  : `various`
        Galactic confusion noise. Acceptable inputs are 'robson19' (the confusion noise from Robson+19),
        `None` (don't include confusion noise) or a custom function that gives the confusion noise at each
        frequency for a given mission length where it would be called by running `noise(f, t_obs)`

    Returns
    -------
    psd : `float/array`
        Effective power strain spectral density
    """
    if instrument == "LISA":
        if L is None:
            L = 2.5e9 * u.m
        psd = lisa_psd(f=f, L=L, t_obs=t_obs, approximate_R=approximate_R, confusion_noise=confusion_noise)
    elif instrument == "TianQin":
        if L is None:
            L = np.sqrt(3) * 1e5 * u.km
        psd = tianqin_psd(f=f, L=L, t_obs=t_obs, approximate_R=approximate_R, confusion_noise=confusion_noise)
    elif instrument == "custom":
        psd = custom_psd(f=f, L=L, t_obs=t_obs, approximate_R=approximate_R, confusion_noise=confusion_noise)
    else:
        raise ValueError("instrument: `{}` not recognised".format(instrument))

    return psd
