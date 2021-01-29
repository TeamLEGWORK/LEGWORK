"""`circular binary stuff!`"""
from astropy import units as u
import numpy as np

import calcs.utils as utils
import calcs.snr as sn

__all__ = ['Source', 'Stationary', 'Evolving']

class Source():
    """Superclass for generic sources"""
    def __init__(self, m_1, m_2, f_orb, ecc, dist, ecc_tol=0.1, stat_tol=1e-2):
        """Initialise all parameters
        Params
        ------
        m_1 : `float/array`
            primary mass

        m_2 : `float/array`
            secondary mass

        forb : `float/array`
            orbital frequency

        ecc : `float/array`
            initial eccentricity

        dist : `float/array`
            luminosity distance to source

        ecc_tol : `float`
            eccentricity above which a binary should be
            considered eccentric

        stat_tol : `float`
            fractional change in frequency above which a
            binary should be considered to be stationary
        """
        self.m_1 = m_1
        self.m_2 = m_2
        self.f_orb = f_orb
        self.ecc = ecc
        self.dist = dist
        self.ecc_tol = ecc_tol
        self.stat_tol = stat_tol
        self.n_sources = len(m_1)

    def get_source_mask(self, circular=None, stationary=None, t_obs=4 * u.yr):
        """Produce a mask of the sources based on whether binaries
        are circular or eccentric and stationary or evolving.
        Tolerance levels are defined in the class.

        Params
        ------
        circular : `bool`
            `None` means either, `True` means only circular
            binaries and `False` means only eccentric

        stationary : `bool`
            `None` means either, `True` means only stationary
            binaries and `False` means only evolving

        t_obs : `float`
            observation time

        Returns
        -------
        mask : `bool/array`
            mask for the sources
        """
        if circular is None:
            circular_mask = np.repeat(True, self.n_sources)
        elif circular is True:
            circular_mask = self.ecc <= self.ecc_tol
        elif circular is False:
            circular_mask = self.ecc > self.ecc_tol
        else:
            raise ValueError("`circular` must be None, True or False")

        if stationary is None:
            stationary_mask = np.repeat(True, self.n_sources)
        elif stationary is True:
            stationary_mask = utils.determine_stationarity(self.m_1, self.m_2,
                                                           self.f_orb, t_obs,
                                                           self.ecc, self.stat_tol)
        elif stationary is False:
            stationary_mask = np.logical_not(utils.determine_stationarity(self.m_1, self.m_2,
                                                           self.f_orb, t_obs,
                                                           self.ecc, self.stat_tol))
        else:
            raise ValueError("`stationary` must be None, True or False")

        return np.logical_and(circular_mask, stationary_mask)

    def get_snr(self, t_obs=4 * u.yr, max_harmonic=50, n_step=100):
        """Computes the SNR for a generic binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        ecc_tol : `float`
            tolerance for treating a binary as eccentric

        stat_tol : `float`
            tolerance for treating a binary as stationary
            (using fractional change in frequency)

        max_harmonic : `int`
            maximum integer harmonic to consider for eccentric sources

        n_step : `int`
            number of time steps during observation duration

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        
        snr = np.zeros(self.n_sources)
        stationary_mask = self.get_source_mask(circular=None, stationary=True, t_obs=t_obs)
        evolving_mask = np.logical_not(stationary_mask)

        if stationary_mask.any():
            snr[stationary_mask] = self.get_snr_stationary(t_obs=t_obs,
                                                           ecc_tol=self.ecc_tol,
                                                           max_harmonic=50,
                                                           which_sources=stationary_mask)
        if evolving_mask.any():
            snr[evolving_mask] = self.get_snr_evolving(t_obs=t_obs,
                                                       ecc_tol=self.ecc_tol,
                                                       max_harmonic=50,
                                                       which_sources=evolving_mask,
                                                       n_step=n_step)
        return snr

    def get_snr_stationary(self, t_obs=4 * u.yr, ecc_tol=0.1,
                           max_harmonic=50, which_sources=None):
        """Computes the SNR assuming a stationary binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        ecc_tol : `float`
            tolerance for treating a binary as eccentric

        max_harmonic : `int`
            maximum integer harmonic to consider for eccentric sources

        which_sources : `bool/array`
            mask on which sources to consider stationary and calculate
            (default is all sources in Class)

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)

        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        snr = np.zeros(self.n_sources)
        ind_ecc = np.logical_and(self.ecc > ecc_tol, which_sources)
        ind_circ = np.logical_and(self.ecc <= ecc_tol, which_sources)

        # only compute snr if there is at least one binary in mask
        if ind_circ.any():
            snr[ind_circ] = sn.snr_circ_stationary(m_c=m_c[ind_circ], 
                                                   f_orb=self.f_orb[ind_circ], 
                                                   dist=self.dist[ind_circ], 
                                                   t_obs=t_obs)
        if ind_ecc.any():
            snr[ind_ecc] = sn.snr_ecc_stationary(m_c=m_c[ind_ecc],
                                                 f_orb=self.f_orb[ind_ecc],
                                                 ecc=self.ecc[ind_ecc],
                                                 dist=self.dist[ind_ecc],
                                                 t_obs=t_obs,
                                                 max_harmonic=max_harmonic)

        return snr[which_sources]

    def get_snr_evolving(self, t_obs, ecc_tol=0.1, max_harmonic=50,
                         n_step=100, which_sources=None):
        """Computes the SNR assuming an evolving binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        ecc_tol : `float`
            tolerance for treating a binary as eccentric

        max_harmonic : `int`
            maximum integer harmonic to consider for eccentric sources

        n_step : `int`
            number of time steps during observation duration

        which_sources : `bool/array`
            mask on which sources to consider evolving and calculate
            (default is all sources in Class)

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
        snr = np.zeros(self.n_sources)
        
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        ind_ecc = np.logical_and(self.ecc > ecc_tol, which_sources)
        ind_circ = np.logical_and(self.ecc <= ecc_tol, which_sources)

        if ind_circ.any():
            snr[ind_circ] = sn.snr_circ_evolving(m_1=self.m_1[ind_circ],
                                                 m_2=self.m_2[ind_circ],
                                                 f_orb_i=self.f_orb[ind_circ],
                                                 dist=self.dist[ind_circ],
                                                 t_obs=t_obs,
                                                 n_step=n_step)
        if ind_ecc.any():
            snr[ind_ecc] = sn.snr_ecc_evolving(m_1=self.m_1[ind_ecc],
                                               m_2=self.m_2[ind_ecc],
                                               f_orb_i=self.f_orb[ind_ecc],
                                               dist=self.dist[ind_ecc],
                                               ecc=self.ecc[ind_ecc],
                                               max_harmonic=max_harmonic,
                                               t_obs=t_obs,
                                               n_step=n_step)

        return snr[which_sources]

class Stationary(Source):
    """Subclass for sources that are stationary"""

    def get_snr(self, t_obs, ecc_tol=0.1, max_harmonic=50, n_step=100):
        return self.get_snr_stationary(t_obs=t_obs, ecc_tol=ecc_tol,
                                       max_harmonic=max_harmonic)
        
class Evolving(Source):
    """Subclass for sources that are evolving"""

    def get_snr(self, t_obs, ecc_tol=0.1, max_harmonic=50, n_step=100):
        return self.get_snr_evolving(t_obs=t_obs, ecc_tol=ecc_tol,
                                     max_harmonic=max_harmonic, n_step=n_step)