"""Functions to compute LISA sensitivity curve"""

import numpy as np
import astropy.units as u
from scipy.interpolate import splev, splrep
from importlib import resources

__all__ = ['load_transfer_function', 'approximate_transfer_function',
           'power_spectral_density']


def load_transfer_function(f, fstar=19.09e-3):
    """Load in transfer function from file
    
    Load transfer function and interpolate values for a range of frequencies. 
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
        Transfer function at each frequency
    """
    # try to load the values for interpolating R
    try:
        with resources.path(package="legwork", resource="R.npy") as path:
            f_R, R = np.load(path)
    except FileExistsError:                             # pragma: no cover
        print("WARNING: Can't find transfer function file, \
                        using approximation instead")
        return approximate_transfer_function(f, fstar)

    # interpolate the R values in the file
    R_data = splrep(f_R * fstar, R, s=0)

    # use interpolated curve to get R values for supplied f values
    R = splev(f, R_data, der=0)
    return R


def approximate_transfer_function(f, fstar):
    """Approximate LISA transfer function

    Use Eq.9 of Robson+19 to approximate the LISA transfer function.

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    Returns
    -------
    R : `float/array`
        Transfer function at each frequency
    """
    return (3 / 10) / (1 + 0.6 * (f / fstar)**2)


def power_spectral_density(f, t_obs=4*u.yr, L=2.5e9, fstar=19.09e-3,
                           approximate_R=False,
                           include_confusion_noise=True):
    """Calculates the effective LISA power spectral density sensitivity
    curve
    
    Using equations from Robson+19, calculate the effective LISA power spectral
    density sensitivity curve

    Parameters
    ----------
    f : `float/array`
        Frequencies at which to evaluate the sensitivity curve

    t_obs : `float`
        Observation time (default 4 years)

    L : `float`
        LISA arm length in metres (default = 2.5Gm)

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    approximate_R : `boolean`
        Whether to approximate the transfer function (default: no)

    include_confusion_noise  : `boolean`
        Whether to include the Galactic confusion noise (default: yes)

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
    HUGE_NOISE = 1e30

    # overwrite frequencies that outside the range
    f = np.where(np.logical_and(f > MIN_F, f < MAX_F), f, 1e-7)

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

        return 9e-45 * f**(-7/3.) \
            * np.exp(-f**(alpha[ind]) + beta[ind] * f
                     * np.sin(kappa[ind] * f)) \
            * (1 + np.tanh(gamma[ind] * (fk[ind] - f)))

    # calculate transfer function (either exactly or with approximation)
    if approximate_R:
        R = approximate_transfer_function(f, fstar)
    else:
        R = load_transfer_function(f, fstar)

    # work out the confusion noise or just set to 0
    if include_confusion_noise:
        cn = Sc(f, t_obs)
    else:
        cn = np.zeros(len(f)) if isinstance(f, (list, np.ndarray)) else 0.0

    # calculate sensitivity curve
    Sn = (1 / (L**2) * (Poms(f) + 4 * Pacc(f) / (2 * np.pi * f)**4)) / R + cn

    # replace values for bad frequencies (set to extremely high value)
    Sn = np.where(np.logical_and(f >= MIN_F, f <= MAX_F), Sn, HUGE_NOISE)
    return Sn / u.Hz
