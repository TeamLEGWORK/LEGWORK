import numpy as np
import calcs.snr as snr
import calcs.source as source
import calcs.utils as utils
import unittest

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_source_snr(self):
        """check that source calculates snr in correct way"""

        # create random (circular/stationary) binaries
        n_values = 500
        t_obs = 4 * u.yr
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        m_c = utils.chirp_mass(m_1, m_2)
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)

        # compare snr calculated directly with through Source
        snr_direct = snr.snr_circ_stationary(m_c=m_c, f_orb=f_orb,
                                             dist=dist, t_obs=t_obs)
        snr_source = sources.get_snr()

        self.assertTrue(np.allclose(snr_direct, snr_source))

        # repeat the same test for eccentric systems
        ecc = np.random.uniform(sources.ecc_tol, 0.1, n_values)
        sources.ecc = ecc

        snr_direct = snr.snr_ecc_stationary(m_c=m_c, f_orb=f_orb, ecc=ecc,
                                            dist=dist, t_obs=t_obs,
                                            max_harmonic=10)
        snr_source = sources.get_snr()

        self.assertTrue(np.allclose(snr_direct, snr_source))

    def test_subclasses(self):
        # create random (circular/stationary) binaries
        n_values = 500
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)

        # compare snr calculated directly with through Source
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)
        stationary_sources = source.Stationary(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                               ecc=ecc, dist=dist)
        self.assertTrue(np.allclose(sources.get_snr(),
                                    stationary_sources.get_snr()))
