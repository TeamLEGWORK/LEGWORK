import numpy as np
import gw.snr as snr
import gw.source as source
import gw.utils as utils
import unittest

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
