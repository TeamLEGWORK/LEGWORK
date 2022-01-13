import numpy as np
import legwork.utils as utils
import unittest
from scipy import integrate

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_keplers_laws(self):
        """check converters are working properly"""

        n_vals = 10000
        f_orb = 10**(np.random.uniform(-5, -1, n_vals)) * u.Hz
        m_1 = np.random.uniform(0, 50, n_vals) * u.Msun
        m_2 = np.random.uniform(0, 50, n_vals) * u.Msun

        # convert frequency to semi-major axis
        a = utils.get_a_from_f_orb(f_orb, m_1, m_2)

        # convert back to frequency
        should_be_f_orb = utils.get_f_orb_from_a(a, m_1, m_2)

        self.assertTrue(np.allclose(f_orb, should_be_f_orb))

    def test_average_response(self):
        """make sure that response integrals are correct"""
        # based on Babak+2021, the sum of the average responses <F_plus^2 a_plus^2>
        # and <F_cross^2 a_cross^2>, when averaged over the position (theta, phi),
        # polarization (psi) and assuming an optimal (face-on) inclination, is 3/10

        def integrand(theta, phi, psi):
            f_plus_2 = utils.F_plus_squared(theta=theta, phi=phi, psi=psi)
            intgl1 = 1 / (4 * np.pi) * (1 / (2 * np.pi)) * f_plus_2 * np.sin(theta)

            f_cross_2 = utils.F_cross_squared(theta=theta, phi=phi, psi=psi)
            intgl2 = 1 / (4 * np.pi) * (1 / (2 * np.pi)) * f_cross_2 * np.sin(theta)

            intgl = intgl1 + intgl2
            return intgl

        result, error = integrate.nquad(
            integrand,
            [[0, np.pi],       # theta
             [0, 2 * np.pi],   # phi
             [0, 2 * np.pi]])  # psi

        self.assertAlmostEqual(result, 3/10)

        def integrand2(theta, phi, psi, inc):
            f_plus_2 = utils.F_plus_squared(theta=theta, phi=phi, psi=psi)
            a_plus_2 = (1 / 4) * (1 + np.cos(inc)**2)**2
            intgl1 = 1 / (4 * np.pi) * (1 / (2 * np.pi)) * f_plus_2 * np.sin(theta) * a_plus_2 * np.sin(inc)

            f_cross_2 = utils.F_cross_squared(theta=theta, phi=phi, psi=psi)
            a_cross_2 = np.cos(inc)**2
            intgl2 = 1 / (4 * np.pi) * (1 / (2 * np.pi)) * f_cross_2 * np.sin(theta) * a_cross_2 * np.sin(inc)

            return intgl1 + intgl2

        result, error = integrate.nquad(
            integrand2,
            [[0, np.pi],  # theta
             [0, 2 * np.pi],  # phi
             [0, 2 * np.pi],  # psi
             [0, np.pi / 2]])  # inc

        self.assertAlmostEqual(result, 0.12)
