import numpy as np
import legwork.snr as snr
import legwork.utils as utils
import unittest

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_circ_vs_ecc_stationary(self):
        """check whether the eccentric snr equation gives the same results
        as the circular one for circular stationary binaries"""
        n_values = 100000
        m_c = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        t_obs = 4 * u.yr

        snr_circ = snr.snr_circ_stationary(m_c=m_c, f_orb=f_orb,
                                           dist=dist, t_obs=t_obs)
        snr_ecc = snr.snr_ecc_stationary(m_c=m_c, f_orb=f_orb,
                                         ecc=np.zeros_like(f_orb).value,
                                         dist=dist, t_obs=t_obs,
                                         harmonics_required=3)
        self.assertTrue(np.allclose(snr_circ, snr_ecc))

    def test_stat_vs_evol_eccentric_and_harmonics(self):
        """check whether the evolving snr equation gives the same results
        as the stationary one for eccentric stationary binaries. Also whether
        the individual harmonics are done properly"""
        n_values = 100
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        m_c = utils.chirp_mass(m_1, m_2)
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-6, -4, n_values)) * u.Hz
        ecc = np.random.uniform(0.1, 0.15, n_values)
        t_obs = 4 * u.yr

        snr_stat = snr.snr_ecc_stationary(m_c=m_c, ecc=ecc,
                                          f_orb=f_orb, dist=dist,
                                          t_obs=t_obs, harmonics_required=10)
        snr_evol = snr.snr_ecc_evolving(m_1=m_1, m_2=m_2, ecc=ecc,
                                        f_orb_i=f_orb, dist=dist, n_step=100,
                                        t_obs=t_obs, harmonics_required=10)

        snr2_stat_n = snr.snr_ecc_stationary(m_c=m_c, ecc=ecc, f_orb=f_orb,
                                             dist=dist, t_obs=t_obs,
                                             harmonics_required=10,
                                             ret_snr2_by_harmonic=True)
        snr2_evol_n = snr.snr_ecc_evolving(m_1=m_1, m_2=m_2, ecc=ecc,
                                           f_orb_i=f_orb, dist=dist, n_step=100,
                                           t_obs=t_obs, harmonics_required=10,
                                           ret_snr2_by_harmonic=True)

        self.assertTrue(np.allclose(snr2_stat_n.sum(axis=1)**(0.5), snr_stat))
        self.assertTrue(np.allclose(snr2_evol_n.sum(axis=1)**(0.5), snr_evol))

        self.assertTrue(np.allclose(snr_stat, snr_evol, atol=1e-1, rtol=1e-2))
