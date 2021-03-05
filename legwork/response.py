"""Functions that determine the reponse of the LISA detector to a GW source with a given sky position, inclination and polarization"""

import numpy as np
import astropy.units as u
import astropy.constants as c

def D_plus_squared(theta, phi):
    """Required for the detector reponses 
    <F_+^2>, <F_x^2>, <F_+F_x>

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    Returns
    -------
    D_plus_2 : `float/array`
        factor used for response; see eq. 44 of Cornish and Larson (2003)
    """

    term_1 = 158 * np.cos(theta)**2
    term_2 = 7 * np.cos(theta)**2
    term_3 = -162 * np.sin(2 * phi) * (1 + np.cos(theta)**2)**2
    D_plus_2 = (3 / 2048) * (487 + term_1 + term_2 + term_3)

    return D_plus_2


def D_cross_squared(theta, phi):
    """Required for the detector reponses 
    <F_+^2>, <F_x^2>, <F_+F_x>

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    Returns
    -------
    D_cross_2 : `float/array`
        factor used for response; see eq. 44 of Cornish and Larson (2003)
    """

    term_1 = 120 * np.sin(theta)**2
    term_2 = np.cos(theta)**2
    term_3 = 162 * np.sin(2 * phi)**2 * np.cos(theta)**2

    D_cross_2 = (3 / 512) * (term_1 + term_2 + term_3)

    return D_cross_2


def D_plus_D_cross(theta, phi):
    """Required for the detector reponses 
    <F_+^2>, <F_x^2>, <F_+F_x>

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source

    Returns
    -------
    D_plus_cross : `float/array`
        factor used for response; see eq. 44 of Cornish and Larson (2003)
    """

    term_1 = np.cos(theta) * np.sin(2 * phi)
    term_2 = 2 * np.cos(phi)**2 - 1
    term_3 = 1 + np.cos(theta)**2

    D_plus_cross = (243 / 512) * term_1 * term_2 * term_3

    return D_plus_cross


def F_plus_squared(theta, phi, psi):
    """The auto-correlated detector response for the 
    plus polarization

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source
 
    psi : `float/array`
        polarization of the source

    Returns
    -------
    F_plus_2 : `float/array`
    """

    term_1 = np.cos(2 * psi)**2 * D_plus_squared(theta, phi)
    term_2 = -np.sin(4 * psi) * D_plus_D_cross(theta, phi)
    term_3 = np.sin(2 * psi)**2 * D_cross_squared(theta, phi)

    F_plus_2 = (1 / 4) * (term_1 + term_2 + term_3)

    return F_plus_2


def F_cross_squared(theta, phi, psi):
    """The auto-correlated detector response for the 
    cross polarization

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source
 
    psi : `float/array`
        polarization of the source

    Returns
    -------
    F_cross_2 : `float/array`
    """

    term_1 = np.cos(2 * psi)**2 * D_cross_squared(theta, phi)
    term_2 = np.sin(4 * psi) * D_plus_D_cross(theta, phi)
    term_3 = np.sin(2 * psi)**2 * D_plus_squared(theta, phi)

    F_cross_2 = (1 / 4) * (term_1 + term_2 + term_3)


def F_plus_F_cross(theta, phi, psi):
    """The cross-correlated detector response for the 
    plus and cross polarizations

    Parameters
    ----------
    theta : `float/array`
        declination of the source

    phi : `float/array`
        right ascension of the source
 
    psi : `float/array`
        polarization of the source

    Returns
    -------
    F_plus_cross : `float/array`
    """

    D_diff = D_plus_squared(theta, phi) - D_cross_squared(theta, phi)
    term_1 = np.sin(4 * psi) * D_diff
    term_2 = 2 * np.cos(4 * psi) * D_plus_D_cross

    F_plus_cross = (1 / 8) * (term_1 + term_2)

    return F_plus_cross
