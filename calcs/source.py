"""`circular binary stuff!`"""
from astropy import units as u
import numpy as np

import calcs.utils as utils
import calcs.snr as sn

__all__ = ['Source', 'Stationary', 'Evolving']

class Source():
    """Superclass for generic sources"""
    def __init__(self, m_1, m_2, f_orb, ecc, dist):
        self.m_1 = m_1
        self.m_2 = m_2
        self.f_orb = f_orb
        self.ecc = ecc
        self.dist = dist
        self.n_sources = len(m_1)

    def get_snr(self, t_obs, ecc_tol=0.1, max_harmonic=50, n_step=100):
        raise NotImplementedError("Haven't done this yet")

    def get_snr_stationary(self, t_obs, ecc_tol=0.1, max_harmonic=50):
        """Computes the SNR assuming a stationary binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        ecc_tol : `float`
            tolerance for treating a binary as eccentric

        max_harmonic : `int`
            maximum integer harmonic to consider for eccentric sources

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
        snr = np.zeros(self.n_sources)
        ind_ecc = self.ecc > ecc_tol
        ind_circ = np.logical_not(ind_ecc)

        # check if everything is circular
        if len(snr) == len(snr[ind_circ]):
            snr = sn.snr_circ_stationary(m_c=m_c, 
                                         f_orb=self.f_orb, 
                                         dist=self.dist, 
                                         t_obs=t_obs)
        # or everything is eccentric
        elif len(snr[ind_circ]) == 0:
            snr = sn.snr_ecc_stationary(m_c=m_c,
                                        f_orb=self.f_orb,
                                        ecc=self.ecc,
                                        dist=self.dist,
                                        t_obs=t_obs,
                                        max_harmonic=max_harmonic)
        # or something in between
        else:
            snr[ind_circ] = sn.snr_circ_stationary(m_c=m_c[ind_circ], 
                                                   f_orb=self.f_orb[ind_circ], 
                                                   dist=self.dist[ind_circ], 
                                                   t_obs=t_obs)

            snr[ind_ecc] = sn.snr_ecc_stationary(m_c=m_c[ind_ecc],
                                                 f_orb=self.f_orb[ind_ecc],
                                                 ecc=self.ecc[ind_ecc],
                                                 dist=self.dist[ind_ecc],
                                                 t_obs=t_obs,
                                                 max_harmonic=max_harmonic)

        return snr.decompose()

class Stationary(Source):
    """Subclass for sources that are stationary"""

    def get_snr(self, t_obs, ecc_tol=0.1, max_harmonic=50, n_step=100):
        return self.get_snr_stationary(t_obs=t_obs, ecc_tol=ecc_tol, max_harmonic=max_harmonic)
        
class Evolving(Source):
    """Subclass for sources that are evolving"""

    def get_snr(self, t_obs, ecc_tol=0.1, max_harmonic=50, n_step=100):
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

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
        snr = np.zeros(self.n_sources)
        ind_ecc, = np.where(self.ecc > ecc_tol)
        ind_circ = np.logical_not(ind_ecc)

        snr[ind_circ] = sn.snr_circ_evolving(m_1=self.m_1[ind_circ],
                                             m_2=self.m_2[ind_circ],
                                             f_orb_i=self.f_orb[ind_circ],
                                             dist=self.dist[ind_circ],
                                             t_obs=t_obs,
                                             n_step=n_step)
        return snr.decompose()