import numpy as np
import calcs.lisa as lisa
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

    def test_confusion_noise(self):
        """check that confusion noise is doing logical things"""
        frequencies = np.logspace(-6, 0, 10000) * u.Hz

        confused = lisa.power_spectral_density(frequencies,
                                               include_confusion_noise=True)
        lucid = lisa.power_spectral_density(frequencies,
                                            include_confusion_noise=False)

        # ensure confusion noise only adds to noise
        self.assertTrue(np.all(confused >= lucid))

        # ensure that it doesn't affect things at low or high frequency
        safe = np.logical_or(frequencies < 1e-4 * u.Hz,
                             frequencies > 1e-2 * u.Hz)
        self.assertTrue(np.allclose(confused[safe], lucid[safe]))

    def test_mission_length_effect(self):
        """check that increasing the mission length isn't changing
        anything far from confusion noise"""
        frequencies = np.logspace(-6, 0, 100) * u.Hz

        # compute same curve with various mission length
        smol = lisa.power_spectral_density(frequencies, t_obs=0.5 * u.yr)
        teeny = lisa.power_spectral_density(frequencies, t_obs=1.0 * u.yr)
        little = lisa.power_spectral_density(frequencies, t_obs=2.0 * u.yr)
        regular = lisa.power_spectral_density(frequencies, t_obs=4.5 * u.yr)
        looonngg = lisa.power_spectral_density(frequencies, t_obs=10.0 * u.yr)
        noises = [smol, teeny, little, regular, looonngg]

        # ensure that a shorter mission length never decreases the noise
        for noise in noises:
            above = noise > regular
            close = np.isclose(noise, regular, atol=1e-39)
            self.assertTrue(np.logical_or(above, close).all())
