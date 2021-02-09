import numpy as np
import calcs.snr as snr
import calcs.source as source
import calcs.strain as strain
import calcs.utils as utils
import unittest

from astropy import units as u


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_source_snr(self):
        """check that source calculates snr in correct way"""

        # create random (circular/stationary) binaries
        n_values = 500
        t_obs = 4 * u.yr
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        m_c = utils.chirp_mass(m_1, m_2)
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)

        # compare snr calculated directly with through Source
        snr_direct = snr.snr_circ_stationary(m_c=m_c, f_orb=f_orb,
                                             dist=dist, t_obs=t_obs)
        snr_source = sources.get_snr(verbose=True)

        self.assertTrue(np.allclose(snr_direct, snr_source))

        # repeat the same test for eccentric systems
        ecc = np.random.uniform(sources.ecc_tol, 0.1, n_values)
        sources.ecc = ecc

        snr_direct = snr.snr_ecc_stationary(m_c=m_c, f_orb=f_orb, ecc=ecc,
                                            dist=dist, t_obs=t_obs,
                                            max_harmonic=10)
        snr_source = sources.get_snr(verbose=True)

        self.assertTrue(np.allclose(snr_direct, snr_source))

    def test_source_strain(self):
        """check that source calculate strain correctly"""
        n_values = 500
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        m_c = utils.chirp_mass(m_1, m_2)
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist, interpolate_g=False)

        source_strain = sources.get_h_0_n([1, 2, 3])
        true_strain = strain.h_0_n(m_c=m_c, f_orb=f_orb, ecc=ecc,
                                   n=[1, 2, 3], dist=dist)

        self.assertTrue(np.all(source_strain == true_strain))

        source_char_strain = sources.get_h_c_n([1, 2, 3])
        true_char_strain = strain.h_c_n(m_c=m_c, f_orb=f_orb, ecc=ecc,
                                        n=[1, 2, 3], dist=dist)

        self.assertTrue(np.all(source_char_strain == true_char_strain))

    def test_stationary_subclass(self):
        # create random (circular/stationary) binaries
        n_values = 500
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)

        # compare snr calculated directly with through Source
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)
        stationary_sources = source.Stationary(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                               ecc=ecc, dist=dist)
        self.assertTrue(np.allclose(sources.get_snr(verbose=True),
                                    stationary_sources.get_snr(verbose=True)))

    def test_evolving_subclass(self):
        # create random (circular/evolving) binaries
        n_values = 500
        m_1 = np.random.uniform(5, 10, n_values) * u.Msun
        m_2 = np.random.uniform(5, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-1.2, -0.5, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)

        # compare snr calculated directly with through Source
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)
        evolving_sources = source.Evolving(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                           ecc=ecc, dist=dist)
        self.assertTrue(np.allclose(sources.get_snr(verbose=True),
                                    evolving_sources.get_snr(verbose=True)))

    def test_masks(self):
        """checks that the masks are being produced correctly"""
        n_values = 10000
        dist = np.random.uniform(0, 30, n_values) * u.kpc

        # all stationary and circular
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-7, -5, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.0, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                                dist=dist, f_orb=f_orb, interpolate_g=False)
        self.assertTrue(sources.get_source_mask(circular=True,
                                                stationary=True).all())

        # all stationary and eccentric
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-7, -5, n_values)) * u.Hz
        ecc = np.random.uniform(0.1, 0.2, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                                dist=dist, f_orb=f_orb, interpolate_g=False)
        self.assertTrue(sources.get_source_mask(circular=False,
                                                stationary=True).all())

        # all evolving and circular
        m_1 = np.random.uniform(5, 10, n_values) * u.Msun
        m_2 = np.random.uniform(5, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-1, 0, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.0, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                                dist=dist, f_orb=f_orb, interpolate_g=False)
        self.assertTrue(sources.get_source_mask(circular=True,
                                                stationary=False).all())

        # all evolving and eccentric
        m_1 = np.random.uniform(5, 10, n_values) * u.Msun
        m_2 = np.random.uniform(5, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-1, 0, n_values)) * u.Hz
        ecc = np.random.uniform(0.1, 0.9, n_values)

        sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                                dist=dist, f_orb=f_orb, interpolate_g=False)
        self.assertTrue(sources.get_source_mask(circular=False,
                                                stationary=False).all())

        # check it works fine if you give Nones
        self.assertTrue(sources.get_source_mask(circular=None,
                                                stationary=None).all())

        # check it crashes if you give nonesense input
        no_worries = True
        try:
            sources.get_source_mask(circular="ridiculous input")
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # check it crashes if you give nonesense input
        no_worries = True
        try:
            sources.get_source_mask(stationary="ridiculous input")
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

    def test_interpolated_g(self):
        """checks that the interpolation of g(n,e) is not producing
        any large errors"""
        # create random binaries
        n_values = 50
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.9, n_values)

        # compare snr calculated directly with through Source
        sources_interp = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                       ecc=ecc, dist=dist, interpolate_g=True)

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist, interpolate_g=False)

        interp_snr = sources_interp.get_snr(verbose=True)
        snr = sources.get_snr(verbose=True)

        self.assertTrue(np.allclose(interp_snr, snr, atol=1e-1))

    def test_bad_input(self):
        """checks that Source handles bad input well"""

        n_values = 10
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        ecc = np.random.uniform(0.0, 1.0, n_values)
        dist = np.random.uniform(0, 10, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz

        # try creating sources with no f_orb or a
        no_worries = True
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # try creating sources with no units
        no_worries = True
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                          dist=dist.value, f_orb=f_orb)
        except AssertionError:
            no_worries = False
        self.assertFalse(no_worries)

        # try creating sources with only single source (should be fine)
        no_worries = True
        source.Source(m_1=1 * u.Msun, m_2=1 * u.Msun,
                      ecc=0.1, dist=8 * u.kpc, f_orb=3e-4 * u.Hz)
        self.assertTrue(no_worries)

        # try creating sources with different length arrays
        no_worries = True
        dist = np.append(dist, 8 * u.kpc)
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                          dist=dist, f_orb=f_orb)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)