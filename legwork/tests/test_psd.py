import numpy as np
import legwork.psd as psd
import unittest
from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_R_approximation(self):
        """check that the R approximation works for low frequencies"""
        frequencies = np.logspace(-6, -2, 10000) * u.Hz

        exact = psd.power_spectral_density(frequencies, approximate_R=True)
        approx = psd.power_spectral_density(frequencies, approximate_R=False)

        self.assertTrue(np.allclose(exact, approx))

    def test_confusion_noise(self):
        """check that confusion noise is doing logical things"""
        frequencies = np.logspace(-6, 0, 10000) * u.Hz

        confused = psd.power_spectral_density(frequencies, confusion_noise="robson19")
        lucid = psd.power_spectral_density(frequencies, confusion_noise=None)

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
        smol = psd.power_spectral_density(frequencies, t_obs=0.5 * u.yr)
        teeny = psd.power_spectral_density(frequencies, t_obs=1.0 * u.yr)
        little = psd.power_spectral_density(frequencies, t_obs=2.0 * u.yr)
        regular = psd.power_spectral_density(frequencies, t_obs=4.5 * u.yr)
        looonngg = psd.power_spectral_density(frequencies, t_obs=10.0 * u.yr)
        noises = [smol, teeny, little, regular, looonngg]

        # ensure that a shorter mission length never decreases the noise
        for noise in noises:
            above = noise > regular
            close = np.isclose(noise, regular, atol=1e-39)
            self.assertTrue(np.logical_or(above, close).all())

    def test_alternate_instruments(self):
        """check that changing instruments doesn't break things"""
        frequencies = np.logspace(-6, 0, 100) * u.Hz

        tq = psd.power_spectral_density(frequencies, instrument="TianQin")

        def custom_instrument(f, t_obs, L, approximate_R, confusion_noise):
            return psd.tianqin_psd(f, L * 2, t_obs, approximate_R, confusion_noise)

        custom = psd.power_spectral_density(frequencies, instrument="custom",
                                            custom_psd=custom_instrument,
                                            L=np.sqrt(3) * 1e5 * u.km)

        self.assertTrue(np.all(custom <= tq))

        all_good = True
        try:
            psd.power_spectral_density(frequencies, instrument="nonsense")
        except ValueError:
            all_good = False
        self.assertFalse(all_good)

    def test_custom_confusion_noise(self):
        """ check that using custom confusion noise works """
        frequencies = np.logspace(-6, 0, 100) * u.Hz

        regular = psd.power_spectral_density(frequencies, confusion_noise=None)
        custom = psd.power_spectral_density(frequencies, confusion_noise=lambda f, t: 0)

        self.assertTrue(np.allclose(regular, custom))
