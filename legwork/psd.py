"""Functions to compute various power spectral densities for sensitivity
curves"""

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import splev, splrep
from importlib import resources

__all__ = ['load_response_function', 'approximate_response_function', 'power_spectral_density',
           'lisa_psd', 'tianqin_psd', 'get_confusion_noise', 'get_confusion_noise_robson19',
           'get_confusion_noise_huang20', 'get_confusion_noise_thiele21']


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

    confusion_noise : `various`
        Galactic confusion noise. Acceptable inputs are one of the values listed in
        :meth:`legwork.psd.get_confusion_noise` or a custom function that gives the confusion noise at each
        frequency for a given mission length where it would be called by running `noise(f, t_obs)` and return
        a value with units of inverse Hertz

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

    # calculate response function (either exactly or with approximation)
    fstar = (const.c / (2 * np.pi * L)).to(u.Hz).value
    if approximate_R:
        R = approximate_response_function(f, fstar)
    else:
        R = load_response_function(f, fstar)

    # get confusion noise
    if isinstance(confusion_noise, str) or confusion_noise is None:
        cn = get_confusion_noise(f=f * u.Hz, t_obs=t_obs, model=confusion_noise).value
    else:
        cn = confusion_noise(f, t_obs).value

    L = L.to(u.m).value

    # calculate sensitivity curve (Robson+19 Eq. 12). Note the factor near Pacc is 2(1 + cos^2(f/f*))
    # not just 4, as in Robson Eq.1, since that's an approximation for low frequencies
    psd = (1 / (L**2) * (Poms(f) + 2 * (1 + np.cos(f / fstar)**2) * Pacc(f) / (2 * np.pi * f)**4)) / R + cn

    # replace values for bad frequencies (set to extremely high value)
    psd = np.where(np.logical_and(f >= MIN_F, f <= MAX_F), psd, np.inf)
    return psd / u.Hz


def tianqin_psd(f, L=np.sqrt(3) * 1e5 * u.km, t_obs=5 * u.yr, approximate_R=None, confusion_noise="huang20"):
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
        Observation time (default 5 years)

    approximate_R : `boolean`
        Ignored for this function

    confusion_noise : `various`
        Galactic confusion noise. Acceptable inputs are one of the values listed in
        :meth:`legwork.psd.get_confusion_noise` or a custom function that gives the confusion noise at each
        frequency for a given mission length where it would be called by running `noise(f, t_obs)` and return
        a value with units of inverse Hertz

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

    # get confusion noise
    if isinstance(confusion_noise, str) or confusion_noise is None:
        cn = get_confusion_noise(f=f, t_obs=t_obs, model=confusion_noise)
    else:
        cn = confusion_noise(f, t_obs)
    psd += cn

    return psd.to(u.Hz**(-1))


def power_spectral_density(f, instrument="LISA", custom_psd=None, t_obs="auto", L="auto",
                           approximate_R=False, confusion_noise="auto"):
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
        Observation time (default 4 years for LISA and 5 years for TianQin)

    L : `float`
        LISA arm length in metres

    approximate_R : `boolean`
        Whether to approximate the response function (default: no)

    confusion_noise : `various`
        Galactic confusion noise. Acceptable inputs are either one of the values listed in
        :meth:`legwork.psd.get_confusion_noise`, "auto" (automatically selects confusion noise based on
        `instrument` - 'robson19' if LISA and 'huang20' if TianQin), or a custom function that gives the
        confusion noise at each frequency for a given mission length where it would be called by running
        `noise(f, t_obs)` and return a value with units of inverse Hertz

    Returns
    -------
    psd : `float/array`
        Effective power strain spectral density
    """
    if instrument == "LISA":
        # update any auto values to be instrument specific
        L = 2.5e9 * u.m if L == "auto" else L
        confusion_noise = "robson19" if confusion_noise == "auto" else confusion_noise
        t_obs = 4 * u.yr if t_obs == "auto" else t_obs

        # calculate psd
        psd = lisa_psd(f=f, L=L, t_obs=t_obs, approximate_R=approximate_R, confusion_noise=confusion_noise)
    elif instrument == "TianQin":
        # update any auto values to be instrument specific
        L = np.sqrt(3) * 1e5 * u.km if L == "auto" else L
        confusion_noise = "huang20" if confusion_noise == "auto" else confusion_noise
        t_obs = 5 * u.yr if t_obs == "auto" else t_obs

        # calculate psd
        psd = tianqin_psd(f=f, L=L, t_obs=t_obs, approximate_R=approximate_R, confusion_noise=confusion_noise)
    elif instrument == "custom":
        psd = custom_psd(f=f, L=L, t_obs=t_obs, approximate_R=approximate_R, confusion_noise=confusion_noise)
    else:
        raise ValueError("instrument: `{}` not recognised".format(instrument))

    return psd


def get_confusion_noise_robson19(f, t_obs=4 * u.yr):
    """Calculate the confusion noise using the model from Robson+19 Eq. 14 and Table 1

    Also note that this fit is designed based on LISA sensitivity and so it is likely not sensible to apply
    it to TianQin or other missions.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to calculate the confusion noise, must have units of frequency
    t_obs : `float`, optional
        Mission length, parameters are defined for 0.5, 1, 2 and 4 years, the closest mission length to the
        one inputted will be used. By default 4 years.

    Returns
    -------
    confusion_noise : `float/array`
        The confusion noise at each frequency
    """
    # erase the units for speed
    f = f.to(u.Hz).value

    # parameters from Robson+ 2019 Table 1
    lengths = np.array([0.5, 1.0, 2.0, 4.0]) * u.yr
    alpha = [0.133, 0.171, 0.165, 0.138]
    beta = [243.0, 292.0, 299.0, -221.0]
    kappa = [482.0, 1020.0, 611.0, 521.0]
    gamma = [917.0, 1680.0, 1340.0, 1680.0]
    fk = [2.58e-3, 2.15e-3, 1.73e-3, 1.13e-3]

    # find index of the closest length to inputted observation time
    ind = np.abs(t_obs - lengths).argmin()

    confusion_noise = 9e-45 * f**(-7 / 3.) * np.exp(-f**(alpha[ind]) + beta[ind]
                                                    * f * np.sin(kappa[ind] * f))\
        * (1 + np.tanh(gamma[ind] * (fk[ind] - f))) * u.Hz**(-1)
    return confusion_noise


def get_confusion_noise_huang20(f, t_obs=5 * u.yr):
    """Calculate the confusion noise using the model from Huang+20 Table II. Note that we set the confusion
    noise to be exactly 0 outside of the range [1e-4, 1] Hz as the fits are not designed to be used outside
    of this range.

    Also note that this fit is designed based on TianQin sensitivity and so it is likely not sensible to apply
    it to LISA or other missions.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to calculate the confusion noise, must have units of frequency
    t_obs : `float`, optional
        Mission length, parameters are defined for 0.5, 1, 2, 4 and 5 years, the closest mission length to the
        one inputted will be used. By default 5 years.

    Returns
    -------
    confusion_noise : `float/array`
        The confusion noise at each frequency
    """

    # define the mission lengths and corresponding coefficients
    lengths = np.array([0.5, 1.0, 2.0, 4.0, 5.0]) * u.yr
    coefficients = np.array([
        [-18.6, -18.6, -18.6, -18.6, -18.6],
        [-1.22, -1.13, -1.45, -1.43, -1.51],
        [0.009, -0.945, 0.315, -0.687, -0.71],
        [-1.87, -1.02, -1.19, 0.24, -1.13],
        [0.65, 4.05, -4.48, -0.15, -0.83],
        [3.6, -4.5, 10.8, -1.8, 13.2],
        [-4.6, -0.5, -9.4, -3.2, -19.1]
    ])

    # find the nearest mission length
    ind = np.abs(t_obs - lengths).argmin()

    # start with no confusion noise and add each coefficient using Huang+20 Table 1
    f = f.to(u.Hz).value
    confusion_noise = np.zeros_like(f)
    x = np.log10(f / 1e-3)
    for i in range(len(coefficients)):
        xi = x**i
        ai = coefficients[i][ind]
        confusion_noise += ai * xi
    confusion_noise = 10**(confusion_noise) * u.Hz**(-1/2)

    # remove confusion noise outside of TianQin regime (fit doesn't apply outside of regime)
    confusion_noise[np.logical_or(f < 1e-4, f > 1e0)] = 0.0 * u.Hz**(-1/2)

    # square the result as Huang+20 given the sensitivity not psd
    return confusion_noise**2


def get_confusion_noise_thiele21(f):
    """Calculate the confusion noise using the model from Thiele+20 Eq. 16 and Table 1. This fit uses a
    metallicity-dependent binary fraction.

    Note: This fit only applies to LISA and only when the mission length is 4 years.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to calculate the confusion noise, must have units of frequency

    Returns
    -------
    confusion_noise : `float/array`
        The confusion noise at each frequency
    """
    x = np.log10(f.to(u.Hz).value)
    t_obs = 4 * u.yr
    coefficients = [-540.315450, -554.061115, -233.748834, -44.021504, -3.112634]
    confusion_noise = 10**np.poly1d(coefficients[::-1])(x) * t_obs.to(u.s).value
    return confusion_noise * u.Hz**(-1)


def get_confusion_noise(f, model, t_obs="auto"):
    """Calculate the confusion noise for a particular model

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to calculate the confusion noise, must have units of frequency
    model : str, optional
        Which model to use for the confusion noise. Must be one of 'robson19', 'huang20', 'thiele21' or None.
    t_obs : `float`, optional
        Mission length. Default is 4 years for robson19 and thiele21 and 5 years for huang20.

    Returns
    -------
    confusion_noise : `float/array`
        The confusion noise at each frequency.

    Raises
    ------
    ValueError
        When a model other than those defined above is used.
    """
    if model == "robson19":
        t_obs = 4 * u.yr if t_obs == "auto" else t_obs
        return get_confusion_noise_robson19(f=f, t_obs=t_obs)
    elif model == "huang20":
        t_obs = 5 * u.yr if t_obs == "auto" else t_obs
        return get_confusion_noise_huang20(f=f, t_obs=t_obs)
    elif model == "thiele21":
        if t_obs == 4 * u.yr or t_obs == "auto":
            return get_confusion_noise_thiele21(f=f)
        else:
            error = "Invalid mission length: Thiele+21 confusion noise is only fit for `t_obs=4*u.yr`"
            raise ValueError(error)
    elif model is None:
        f = f.to(u.Hz).value
        return np.zeros_like(f) * u.Hz**(-1) if isinstance(f, (list, np.ndarray)) else 0.0 * u.Hz**(-1)
    else:
        raise ValueError("confusion noise model: `{}` not recognised".format(model))
