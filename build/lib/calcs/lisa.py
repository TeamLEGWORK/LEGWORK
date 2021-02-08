"""Computes LISA sensitivity curve"""

import numpy as np
import astropy.units as u

def power_spec_sens(f_gw):
    """Computes the LISA power spectral density sensitivity curve 
    for the current LISA mission (2020) from Smith & Caldwell (2019)

    Params
    ------
    f : `float/array`
        gravitational wave frequency

    Returns
    -------
    sigma_h : `float/array`
        LISA power spectral density sensitivity
    """
 
    f1 = 0.4e-3
    f2 = 25e-3
    R = 1 + (f_gw/f2)**2

    S_I = 5.76e-48 * (1 + (f1/f_gw)**2)
    S_II = 3.6e-41

    sigma_h = (1/2) * (20/3) * (S_I/(2*np.pi*f_gw)**4 + S_II) * R

    return sigma_h * u.s
