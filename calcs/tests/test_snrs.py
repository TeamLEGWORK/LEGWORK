import numpy as np
import calcs.snr as snr
import unittest

from astropy import units as u

n_values = 100000

m_c = np.random.uniform(0, 10, n_values) * u.Msun
dist = np.random.uniform(0, 30, n_values) * u.kpc
f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
e = np.random.uniform(0, 0.9, n_values)
n = 2

class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_circular_vs_eccentric_snr(self):
        """check whether the eccentric snr equation gives the same results
        as the circular one for circular binaries"""
        t_obs = 4 * u.yr
        snr_circ = snr.snr_circ_stationary(m_c, f_orb, dist, t_obs).decompose()
        snr_ecc = snr.snr_ecc_stationary(m_c, f_orb, 0.0, dist, t_obs, 25).decompose()

        difference = snr_circ - snr_ecc

        self.assertTrue(all(difference.value < 1e-5))
