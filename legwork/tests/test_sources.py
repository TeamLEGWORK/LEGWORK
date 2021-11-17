import numpy as np
import legwork.snr as snr
import legwork.source as source
import legwork.strain as strain
import legwork.utils as utils
import unittest

from astropy import units as u
from astropy.coordinates import SkyCoord


class Test(unittest.TestCase):
    """Tests that the code is functioning properly"""

    def test_evolution_functions(self):
        """Test that evolving sources works as expected"""
        n_values = 50
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        ecc = np.linspace(0.0, 0.4, n_values)

        # compare snr calculated directly with through Source
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist, interpolate_g=False,
                                interpolate_sc=False)

        # calculate the merger times
        t_merge = sources.get_merger_time(save_in_class=False)

        # create a new class after evolving every source for 10 years
        evolved_sources = sources.evolve_sources(10 * u.yr,
                                                 create_new_class=True)

        # evolve one of the evolved sources for a little more time
        t_evol = np.zeros(n_values) * u.yr
        t_evol[0] = 1 * u.yr
        evolved_sources.evolve_sources(t_evol)

        # ensure that merger times have been updated correctly
        final_merger_times = t_merge - (10 * u.yr)
        final_merger_times[0] -= 1 * u.yr
        final_merger_times[final_merger_times < 0 * u.yr] = 0 * u.yr
        self.assertTrue(np.allclose(final_merger_times,
                                    evolved_sources.t_merge))

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
                                            harmonics_required=10)
        snr_source = sources.get_snr(verbose=True)

        self.assertTrue(np.allclose(snr_direct, snr_source))

    def test_source_snr_multi(self):
        """check that source calculates snr in correct way"""

        # create random (circular/stationary) binaries
        n_values = 500
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-3, -2, n_values)) * u.Hz
        ecc = np.random.uniform(0.1, 0.2, n_values)
        n_proc = 2
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist, n_proc=n_proc)

        sources_1 = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                  ecc=ecc, dist=dist, n_proc=1)

        # compare using 1 or 2 processors
        snr_2 = sources.get_snr(verbose=True)
        snr_1 = sources_1.get_snr(verbose=True)

        self.assertTrue(np.allclose(snr_2, snr_1))

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
                                   n=[1, 2, 3], dist=dist)[:, 0, :]

        self.assertTrue(np.all(source_strain == true_strain))

        source_char_strain = sources.get_h_c_n([1, 2, 3])
        true_char_strain = strain.h_c_n(m_c=m_c, f_orb=f_orb, ecc=ecc,
                                        n=[1, 2, 3], dist=dist)[:, 0, :]

        self.assertTrue(np.all(source_char_strain == true_char_strain))

    def test_amplitude_modulation_h_0_n(self):
        """Make sure that the amplitude modulated strains are correct.
        Note that this is very redundant with the utils modulation tests"""
        n_values = 500
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        m_c = utils.chirp_mass(m_1, m_2)
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)
        incs = np.arccos(np.random.uniform(-1, 1, n_values)) * u.rad
        thetas = np.arcsin(np.random.uniform(-1, 1, n_values)) * u.rad
        phis = np.random.uniform(0, 2 * np.pi, n_values) * u.rad
        psis = np.random.uniform(0, 2 * np.pi, n_values) * u.rad

        positions = SkyCoord(phis, thetas, distance=dist, frame='heliocentrictrueecliptic')

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist,
                                position=positions, inclination=incs,
                                polarisation=psis, interpolate_g=False)
        source_strains = sources.get_h_0_n([1, 2, 3])
        true_strain = strain.h_0_n(m_c=m_c, f_orb=f_orb, ecc=ecc,
                                   dist=dist, position=positions,
                                   inclination=incs, polarisation=psis,
                                   n=[1, 2, 3])[:, 0, :]
        self.assertTrue(np.all(source_strains == true_strain))

    def test_amplitude_modulation_h_c_n(self):
        """Make sure that the amplitude modulated strains are correct.
        Note that this is very redundant with the utils modulation tests"""
        n_values = 500
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        m_c = utils.chirp_mass(m_1, m_2)
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -4, n_values)) * u.Hz
        ecc = np.repeat(0.0, n_values)
        incs = np.arccos(np.random.uniform(-1, 1, n_values)) * u.rad
        thetas = np.arcsin(np.random.uniform(-1, 1, n_values)) * u.rad
        phis = np.random.uniform(0, 2 * np.pi, n_values) * u.rad
        psis = np.random.uniform(0, 2 * np.pi, n_values) * u.rad

        positions = SkyCoord(phis, thetas, distance=dist, frame='heliocentrictrueecliptic')

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist,
                                position=positions, inclination=incs,
                                polarisation=psis, interpolate_g=False)
        source_strains = sources.get_h_c_n([1, 2, 3])
        true_strain = strain.h_c_n(m_c=m_c, f_orb=f_orb, ecc=ecc, dist=dist, position=positions,
                                   inclination=incs, polarisation=psis, n=[1, 2, 3])[:, 0, :]
        self.assertTrue(np.all(source_strains == true_strain))

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
        np.random.seed(42)
        n_values = 50
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -3, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.9, n_values)

        # compare snr calculated directly with through Source
        sources_interp = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                       ecc=ecc, dist=dist, interpolate_g=True)

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist, interpolate_g=False)

        interp_snr = sources_interp.get_snr(verbose=True)
        snr = sources.get_snr(verbose=True)

        self.assertTrue(np.allclose(interp_snr, snr, atol=1e-1, rtol=1e-1))

    def test_interpolated_sc(self):
        """checks that interpolated of LISA SC is not producing any large
        errors"""
        # create random binaries
        n_values = 50
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        dist = np.random.uniform(0, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.4, n_values)

        # compare snr calculated directly with through Source
        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb,
                                ecc=ecc, dist=dist)
        interp_snr = sources.get_snr(verbose=True)

        # erase interpolation
        sources.interpolate_sc = False
        sources.update_sc_params(None)

        snr = sources.get_snr(verbose=True)

        self.assertTrue(np.allclose(interp_snr, snr, atol=1e-1, rtol=1e-1))

    def test_bad_input(self):
        """checks that Source handles bad input well"""

        n_values = 10
        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        ecc = np.random.uniform(0.0, 0.95, n_values)
        dist = np.random.uniform(0, 10, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        position = SkyCoord(lat=np.random.uniform(0.0, 90, n_values) * u.deg,
                            lon=np.random.uniform(0, 360, n_values) * u.deg,
                            distance=dist, frame="heliocentrictrueecliptic")
        inclination = np.arcsin(np.random.uniform(-1, 1, n_values)) * u.rad
        polarisation = np.random.uniform(0, 2 * np.pi, n_values) * u.rad

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

        # try creating sources with only single source with some in arrays
        no_worries = True
        source.Source(m_1=1 * u.Msun, m_2=1 * u.Msun,
                      ecc=[0.1], dist=8 * u.kpc, f_orb=3e-4 * u.Hz)
        self.assertTrue(no_worries)

        # try creating a source with inclination but not position
        no_worries = True
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                          dist=dist, f_orb=f_orb, inclination=inclination)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # try creating a source with polarisation but not position
        no_worries = True
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                          dist=dist, f_orb=f_orb, polarisation=polarisation)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # create a source with position but not inclination or polarisation with eccentric sources
        no_worries = True
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                          dist=dist, f_orb=f_orb, position=position)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # create a source with position but not inclination or polarisation with circular sources
        ecc = np.zeros_like(ecc)
        no_worries = True
        try:
            source.Source(m_1=m_1, m_2=m_2, ecc=ecc,
                          dist=dist, f_orb=f_orb, position=position)
        except ValueError:
            no_worries = False
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

    def test_verification_binaries(self):
        """simple to test to check if you can instantiate VerificationBinaries"""

        no_worries = True
        try:
            source.VerificationBinaries()
        except ValueError:
            no_worries = False
        self.assertTrue(no_worries)

    def test_updating_sc_params(self):
        """ ensuring that updating the sc params always works """
        original_sc_params = {
            "instrument": "LISA",
            "t_obs": 4 * u.yr,
            "L": 2.5e9 * u.m,
            "approximate_R": False,
            "confusion_noise": "robson19"
        }

        sources = source.Source(m_1=1 * u.Msun, m_2=1 * u.Msun, f_orb=1e-3 * u.Hz, ecc=0.2, dist=10*u.kpc,
                                sc_params=original_sc_params)

        sources.update_sc_params({"instrument": "TianQin", "L": np.sqrt(3) * 1e5 * u.km})

        correct_final_sc_params = {
            "instrument": "TianQin",
            "t_obs": 4 * u.yr,
            "L": np.sqrt(3) * 1e5 * u.km,
            "approximate_R": False,
            "confusion_noise": "robson19",
            "custom_psd": None,
        }
        self.assertTrue(correct_final_sc_params == sources._sc_params)
