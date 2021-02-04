"""Computes LISA sensitivity curve"""

import numpy as np
import astropy.units as u
from scipy.interpolate import splev, splrep
from importlib import resources


def load_transfer_function(f, fstar=19.09e-3):
    """Load in transfer function from file and interpolate values
    for a range of frequencies. Adapted from
    https://github.com/eXtremeGravityInstitute/LISA_Sensitivity to use binary
    files instead of text. See Robson+19 for more details

    Params
    ------
    f : `float/array`
        frequencies at which to evaluate the sensitivity curve

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    Returns
    -------
    R : `float/array`
        transfer function at each frequency
    """
    # try to load the values for interpolating R
    try:
        with resources.path(package="calcs", resource="R.npy") as path:
            f_R, R = np.load(path)
    except FileExistsError:
        print("WARNING: Can't find transfer function file, using approximation instead")
        return approximate_transfer_function(f, fstar)

    # interpolate the R values in the file
    R_data = splrep(f_R * fstar, R, s=0)

    # use interpolated curve to get R values for supplied f values
    R = splev(f, R_data, der=0)
    return R


def approximate_transfer_function(f, fstar):
    """Calculate the the LISA transfer function using
    approximation from Eq. 9 of Robson+19

    Params
    ------
    f : `float/array`
        frequencies at which to evaluate the sensitivity curve

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    Returns
    -------
    R : `float/array`
        transfer function at each frequency
    """
    return (3 / 10) / (1 + 0.6 * (f / fstar)**2)


def power_spectral_density(f, t_obs=4*u.yr, L=2.5e9, fstar=19.09e-3, approximate_R=False, include_confusion_noise=True):
    """Calculates the effective LISA power spectral density sensitivity
    curve using equations from Robson+19

    Params
    ------
    f : `float/array`
        frequencies at which to evaluate the sensitivity curve

    t_obs : `float`
        observation time (default 4 years)

    L : `float`
        LISA arm length in metres (default = 2.5Gm)

    fstar : `float`
        f* from Robson+19 (default = 19.09 mHz)

    approximate_R : `boolean`
        whether to approximate the transfer function (default: no)
    
    include_confusion_noise  : `boolean`
        whether to include the Galactic confusion noise (default: yes)
    
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

        # use parameters from Robson+ 2019 by finding closest number of years
        years = t_obs.to(u.yr).value
        if years < 0.75:
            alpha = 0.133
            beta = 243.
            kappa = 482.
            gamma = 917.
            fk = 2.58e-3
        elif years < 1.5:
            alpha = 0.171
            beta = 292.
            kappa = 1020.
            gamma = 1680.
            fk = 2.15e-3
        elif years < 3.0:
            alpha = 0.165
            beta = 299.
            kappa = 611.
            gamma = 1340.
            fk = 1.73e-3
        else:
            alpha = 0.138
            beta = -221.
            kappa = 521.
            gamma = 1680.
            fk = 1.13e-3

        return 9e-45 * f**(-7/3.) * np.exp(-f**(alpha) + beta * f \
                * np.sin(kappa * f)) * (1 + np.tanh(gamma * (fk - f)))

    # calculate transfer function (either exactly or with approximation)
    if approximate_R:
        R = approximate_transfer_function(f, fstar)
    else:
        R = load_transfer_function(f, fstar)

    # work out the confusion noise or just set to 0
    if include_confusion_noise:
        cn = Sc(f, t_obs)
    else:
        cn = np.zeros(len(f)) if isinstance(f, list) or isinstance(f, np.ndarray) else 0.0

    # calculate sensitivity curve
    Sn = (1 / (L**2) * (Poms(f) + 4 * Pacc(f) / (2 * np.pi * f)**4)) / R  + cn

    # replace values for bad frequencies (set to extremely high value)
    Sn = np.where(np.logical_and(f > MIN_F, f < MAX_F), Sn, HUGE_NOISE)
    return Sn / u.Hz
