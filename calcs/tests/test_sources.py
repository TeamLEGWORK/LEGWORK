import numpy as np
import calcs.snr as snr
import calcs.source as source
import calcs.utils as utils
import unittest

from astropy import units as u

class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_strain_conversion(self):
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

        # compare snr calculated directly with through Source
        true_snr = snr.snr_circ_stationary(m_c=m_c, f_orb=f_orb, dist=dist, t_obs=t_obs)
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)
        difference = true_snr - sources.get_snr()
        self.assertTrue(all(difference.value < 1e-5))

        # repeat the same test for eccentric systems
        ecc = np.random.uniform(0, 0.05, n_values)

        true_snr = snr.snr_ecc_stationary(m_c=m_c, f_orb=f_orb, ecc=ecc, dist=dist, t_obs=t_obs, max_harmonic=50)
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist, ecc_tol=1e-5)
        difference = true_snr - sources.get_snr()

        self.assertTrue(all(difference.value < 1e-5))