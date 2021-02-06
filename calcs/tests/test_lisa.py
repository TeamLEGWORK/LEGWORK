import numpy as np
import calcs.lisa as lisa
import calcs.utils as utils
import unittest

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_R_approximation(self):
        """check that the R approximation works for low frequencies"""
        frequencies = np.logspace(-6, -2, 10000) * u.Hz

        exact = lisa.power_spectral_density(frequencies, approximate_R=True)
        approx = lisa.power_spectral_density(frequencies, approximate_R=False)

        self.assertTrue(np.allclose(exact, approx))
