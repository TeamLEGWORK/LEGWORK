import numpy as np
import legwork.strain as strain
import legwork.utils as utils
import unittest
from astropy.coordinates import SkyCoord

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_strain_conversion(self):
        """This test checks whether the strain and characteristic strain are
        related as hc^2 = (fn^2 / fn_dot) h0^2"""
        n_values = 100000

        m_c = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        e = np.random.uniform(0, 0.9, n_values)
        n = 2

        h0 = strain.h_0_n(m_c, f_orb, e, n, dist).flatten()
        hc = strain.h_c_n(m_c, f_orb, e, n, dist).flatten()

        should_be_fn_dot = (n * f_orb)**2 * h0**2 / hc**2
        fn_dot = utils.fn_dot(m_c, f_orb, e, n)

        self.assertTrue(np.allclose(should_be_fn_dot, fn_dot))

    def test_modulation_face_on(self):
        """Make sure that the modulation is consistent for edge on binaries"""
        dist = 100 * u.parsec
        inclination = np.pi / 2 * u.rad
        polarization = 0.0 * u.rad
        lat = np.pi / 2 * u.rad
        lon = np.pi / 2 * u.rad

        position = SkyCoord(lat=lat, lon=lon, distance=dist, frame='heliocentrictrueecliptic')

        # the 4/5 here is because we undo the total averaging in the modulation
        mod = 4 / 5 * strain.amplitude_modulation(position, polarization, inclination)

        F_plus_squared = utils.F_plus_squared(theta=lat, phi=lon, psi=polarization)

        self.assertEqual(mod, F_plus_squared / 2)

    def test_modulation_edge_on(self):
        """Make sure that the modulation is consistent for edge on binaries"""
        dist = 100 * u.parsec
        inclination = 0 * u.rad
        polarization = 0.0 * u.rad
        lat = np.pi / 2 * u.rad
        lon = np.pi / 2 * u.rad

        position = SkyCoord(lat=lat, lon=lon, distance=dist, frame='heliocentrictrueecliptic')

        # the 4/5 here is because we undo the total averaging in the modulation
        mod = 4 / 5 * strain.amplitude_modulation(position, polarization, inclination)

        F_plus_squared = utils.F_plus_squared(theta=lat, phi=lon, psi=polarization)
        F_cross_squared = utils.F_cross_squared(theta=lat, phi=lon, psi=polarization)

        mod_by_hand = 0.5 * ((1 + np.cos(inclination)**2)**2 * F_plus_squared +
                             4 * np.cos(inclination)**2 * F_cross_squared)
        self.assertEqual(mod, mod_by_hand)
