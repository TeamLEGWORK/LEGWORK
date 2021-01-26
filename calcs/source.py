"""`circular binary stuff!`"""
from astropy import units as u
import numpy as np

import calcs.utils as utils
import calcs.snr as sn

__all__ = ['Stationary', 'Evolving']


class Stationary:
    """Treats stationary sources"""

    def __init__(self, m_1, m_2, f_orb, ecc, dist):
        """Initialize!"""
        self.m_1 = m_1 * u.Msun
        self.m_2 = m_2 * u.Msun
        self.f_orb = f_orb * u.s**(-1)
        self.ecc = ecc
        self.dist = dist * u.kpc

    def get_snr(self, t_obs, ecc_tol=0.1, n_max=50):
        """Computes the SNR assuming a stationary binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        ecc_tol : `float`
            tolerance for treating a binary as eccentric

        n_max : `int`
            maximum integer harmonic to consider for eccentric sources

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        t_obs = t_obs * u.yr
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
        snr = np.zeros(len(self.m_1))
        ind_ecc, = np.where(self.ecc > ecc_tol)
        ind_circ, = np.where(self.ecc <= ecc_tol)

        #Treat circular first
        snr[ind_circ] = sn.snr_circ_stationary(m_c=m_c[ind_circ].to(u.kg), 
                                               f_orb=self.f_orb[ind_circ], 
                                               dist=self.dist[ind_circ].to(u.m), 
                                               t_obs=t_obs.to(u.s))

        snr[ind_ecc] = sn.snr_ecc_stationary(m_c=m_c[ind_ecc].to(u.kg),
                                             f_orb=self.f_orb[ind_ecc],
                                             ecc=self.ecc[ind_ecc],
                                             dist=self.dist[ind_ecc].to(u.m),
                                             t_obs=t_obs.to(u.s),
                                             n_max=n_max)

        return snr

class Evolving:
    """Treats evolving sources"""

    def __init__(self, m_1, m_2, f_orb, ecc, dist):
        """Initialize!"""
        self.m_1 = m_1 * u.Msun
        self.m_2 = m_2 * u.Msun
        self.f_orb = f_orb * u.s**(-1)
        self.dist = dist * u.kpc
        self.ecc = ecc

    def get_snr(self, t_obs, ecc_tol=0.1, n_max=50, n_step=100):
        """Computes the SNR assuming an evolving binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        ecc_tol : `float`
            tolerance for treating a binary as eccentric

        n_max : `int`
            maximum integer harmonic to consider for eccentric sources

        n_step : `int`
            number of timesteps during observation duration

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        t_obs = t_obs * u.yr
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
        snr = np.zeros(len(self.m_1))
        ind_ecc, = np.where(self.ecc > ecc_tol)
        ind_circ, = np.where(self.ecc <= ecc_tol)

        snr[ind_circ] = sn.snr_circ_evolving(m_1=self.m_1[ind_circ].to(u.kg),
                                             m_2=self.m_2[ind_circ].to(u.kg),
                                             f_orb_i=self.f_orb[ind_circ],
                                             dist=self.dist[ind_circ].to(u.m),
                                             t_obs=t_obs.to(u.s),
                                             n_step=n_step)
        return snr

