import numpy as np
import calcs.evol as evol
import unittest

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_upper_bound(self):
        """checks that the circular merger time is an upper bound for the
        true merger time"""
        n_values = 10000

        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        e = np.random.uniform(0, 1, n_values)

        circ_time = evol.get_t_merge_circ(m_1=m_1, m_2=m_2, f_orb_i=f_orb)
        ecc_time = evol.get_t_merge_ecc(m_1=m_1, m_2=m_2,
                                        f_orb_i=f_orb, ecc_i=e)

        self.assertTrue(np.all(circ_time >= ecc_time))

    def test_circular_case(self):
        """checks that you get the same value if all binaries are
        exactly circular"""
        n_values = 10000

        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        e = np.repeat(0.0, n_values)

        circ_time = evol.get_t_merge_circ(m_1=m_1, m_2=m_2, f_orb_i=f_orb)
        ecc_time = evol.get_t_merge_ecc(m_1=m_1, m_2=m_2,
                                        f_orb_i=f_orb, ecc_i=e)

        self.assertTrue(np.allclose(circ_time, ecc_time))

    def test_t_merge_circ_bad_input(self):
        """checks that the t_merge circ function handles bad input well"""
        # missing masses
        no_worries = True
        try:
            evol.get_t_merge_circ(beta=None, m_1=None, m_2=None,
                                  f_orb_i=1e-3 * u.Hz)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing individual masses and no a_i
        no_worries = True
        try:
            evol.get_t_merge_circ(beta=10 * u.m**4 / u.s,
                                  f_orb_i=1e-3 * u.Hz, a_i=None)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing frequency and separation
        no_worries = True
        try:
            evol.get_t_merge_circ(m_1=10 * u.Msun, m_2=10 * u.Msun,
                                  f_orb_i=None, a_i=None)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

    def test_t_merge_ecc_bad_input(self):
        """checks that the t_merge ecc function handles bad input well"""
        # missing masses
        no_worries = True
        try:
            evol.get_t_merge_ecc(beta=None, m_1=None, m_2=None,
                                 f_orb_i=1e-3 * u.Hz, ecc_i=0.0)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing individual masses and no a_i
        no_worries = True
        try:
            evol.get_t_merge_ecc(beta=10 * u.m**4 / u.s,
                                 f_orb_i=1e-3 * u.Hz, a_i=None, ecc_i=0.0)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing frequency and separation
        no_worries = True
        try:
            evol.get_t_merge_ecc(m_1=10 * u.Msun, m_2=10 * u.Msun,
                                 f_orb_i=None, a_i=None, ecc_i=0.0)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)
