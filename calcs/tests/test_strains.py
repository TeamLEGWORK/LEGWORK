import numpy as np
import calcs.strain as strain
import calcs.utils as utils
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

    def test_strain_conversion(self):
        """This test checks whether the strain and characteristic strain are
        related as hc^2 = (2 * fn^2 / fn_dot) h0^2"""
        h0 = strain.h_0_n(m_c, f_orb, e, n, dist)
        hc = strain.h_c_n(m_c, f_orb, e, n, dist)

        should_be_fn_dot = 2 * (n * f_orb)**2 * h0**2 / hc**2
        fn_dot = utils.fn_dot(m_c, f_orb, e, n)

        difference = should_be_fn_dot.to(u.Hz / u.yr).round(10) - fn_dot.to(u.Hz / u.yr).round(10)

        self.assertTrue(all(difference.value < 1e-5))