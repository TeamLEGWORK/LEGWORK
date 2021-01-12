"""`circular binary stuff!`"""
from astropy import units as u
import calcs.utils as utils
import calcs.strain as strain
import calcs.lisa as lisa

__all__ = ['Stationary']


class Stationary:
    """Treats stationary sources"""

    def __init__(self, m_1, m_2, f_orb, ecc, dist):
        """Initialize baby!"""
        self.m_1 = m_1 * u.Msun
        self.m_2 = m_2 * u.Msun
        self.f_orb = f_orb * u.s**(-1)
        self.dist = dist * u.kpc
        self.ecc = ecc

    def get_snr(self, t_obs):
        """Computes the SNR assuming a stationary circular binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        t_obs = t_obs * u.yr
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
        if self.ecc.all() == 0.0:
            #only need to compute n=2
            h_0 = strain.h_0_n(m_c = m_c.to(u.kg),
                              f_orb=self.f_orb, ecc=self.ecc,
                              n=2, dist=self.dist.to(u.m))

            h_f_src = h_0 * (t_obs.to(u.s))**0.5
            h_f_lisa = lisa.power_spec_sens(f_gw = (2*self.f_orb)/u.s**(-1))
            snr = h_f_src / (2*h_f_lisa**0.5)

        return snr
