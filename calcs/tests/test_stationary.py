import numpy as np
import calcs.source as source
import unittest

m1 = np.array([1.0])
m2 = np.array([1.0])
f_orb = np.array([1/3600.0])
dist = np.array([1.0])
t_obs = np.array([4.0])
ecc = np.array([0.0])

SNR_CIRC_STAT = np.array([10.21795672]) / 2

STATIONARYCLASS = source.Stationary(m_1=m1, m_2=m2, f_orb=f_orb, dist=dist, ecc=ecc)


class TestStationary(unittest.TestCase):
    """`TestCasei` for computing the SNR of stationary binaries"""

    def test_snr_circular(self):
        snr_circ = STATIONARYCLASS.get_snr(t_obs=t_obs)
        self.assertEqual(np.round(snr_circ[0], 8), SNR_CIRC_STAT[0])
