import h5py
import io
import numpy as np
import legwork.psd as psd
import legwork.snr as snr
import legwork.source as source
import legwork.strain as strain
import legwork.utils as utils
import os
import tempfile
import unittest

from astropy import units as u
from astropy.coordinates import SkyCoord
from contextlib import redirect_stdout


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

        # the semi-major axis should follow the evolved frequencies rather than going stale
        self.assertTrue(np.allclose(evolved_sources.a,
                                    utils.get_a_from_f_orb(evolved_sources.f_orb, m_1, m_2)))

    def test_evolving_circular_sources(self):
        """check that exactly circular sources are evolved (they take a different path to eccentric ones)"""
        n_values = 10

        # use wide, low mass binaries so that nothing merges during the evolution
        m_1 = np.repeat(1, n_values) * u.Msun
        m_2 = np.repeat(1, n_values) * u.Msun
        dist = np.repeat(10, n_values) * u.kpc
        f_orb = np.repeat(1e-4, n_values) * u.Hz

        # a mixture of exactly circular and eccentric sources exercises both branches at once
        ecc = np.repeat(0.0, n_values)
        ecc[n_values // 2:] = 0.3

        sources = source.Source(m_1=m_1, m_2=m_2, f_orb=f_orb, ecc=ecc, dist=dist,
                                interpolate_g=False, interpolate_sc=False)
        evolved = sources.evolve_sources(1000 * u.yr, create_new_class=True)

        # nothing should have merged and every source should have been evolved to a higher frequency
        self.assertFalse(evolved.merged.any())
        self.assertTrue(np.all(evolved.f_orb > f_orb))

        # the circular sources should still be circular, the eccentric ones should have circularised
        self.assertTrue(np.all(evolved.ecc[:n_values // 2] == 0.0))
        self.assertTrue(np.all(evolved.ecc[n_values // 2:] < ecc[n_values // 2:]))

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

        positions = SkyCoord(phis, thetas, distance=dist, frame='barycentrictrueecliptic')

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

        positions = SkyCoord(phis, thetas, distance=dist, frame='barycentrictrueecliptic')

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
        sources.sc_params = None

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
            vbs = source.VerificationBinaries()
        except ValueError:
            no_worries = False
        self.assertTrue(no_worries)

        # the verification binaries are a fixed set of sources so they can't be subsetted
        it_broke = False
        try:
            vbs[0]
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_updating_sc_params(self):
        """ ensuring that updating the sc params always works """
        original_sc_params = {
            "instrument": "LISA",
            "custom_psd": None,
            "t_obs": "auto",
            "L": "auto",
            "approximate_R": False,
            "confusion_noise": "auto"
        }

        sources = source.Source(m_1=1 * u.Msun, m_2=1 * u.Msun, f_orb=1e-3 * u.Hz, ecc=0.2, dist=10*u.kpc,
                                sc_params=original_sc_params)

        # assigning a new set of params should reset anything that isn't supplied to its default
        sources.sc_params = {"instrument": "TianQin"}

        correct_final_sc_params = {
            "instrument": "TianQin",
            "t_obs": "auto",
            "L": "auto",
            "approximate_R": False,
            "confusion_noise": "auto",
            "custom_psd": None,
        }
        self.assertTrue(correct_final_sc_params == sources.sc_params)

    def test_sc_params_re_interpolation(self):
        """check that the sensitivity curve is re-interpolated when the params change"""
        sources = source.Source(m_1=1 * u.Msun, m_2=1 * u.Msun, f_orb=1e-3 * u.Hz, ecc=0.0, dist=10 * u.kpc,
                                interpolate_g=False, sc_params={"instrument": "LISA"})

        # changing a single value should re-interpolate the curve
        original_sc = sources.sc
        original_value = sources.sc(1e-3 * u.Hz)
        sources.sc_params["instrument"] = "TianQin"

        self.assertTrue(sources.sc_params["instrument"] == "TianQin")
        self.assertTrue(sources.sc is not original_sc)
        # (atol=0 since the PSD values are tiny, so everything is "close" by default)
        self.assertFalse(np.isclose(sources.sc(1e-3 * u.Hz), original_value, atol=0))

        # everything else should be left alone
        self.assertTrue(sources.sc_params["t_obs"] == "auto")

        # setting the same value again shouldn't bother re-interpolating
        unchanged_sc = sources.sc
        sources.sc_params["instrument"] = "TianQin"
        self.assertTrue(sources.sc is unchanged_sc)

        # changing several values at once should only re-interpolate once
        n_interpolations = [0]
        real_set_sc = sources.set_sc

        def counting_set_sc():
            n_interpolations[0] += 1
            real_set_sc()
        sources.set_sc = counting_set_sc

        sources.sc_params.update({"instrument": "LISA", "t_obs": 2 * u.yr})
        self.assertTrue(n_interpolations[0] == 1)
        self.assertTrue(sources.sc_params["t_obs"] == 2 * u.yr)

    def test_mismatched_sc_params(self):
        """check that params passed to the SNR functions are matched by the interpolated curve"""
        n_values = 5
        args = {"m_1": np.repeat(1, n_values) * u.Msun, "m_2": np.repeat(1, n_values) * u.Msun,
                "dist": np.repeat(10, n_values) * u.kpc, "ecc": np.zeros(n_values),
                "f_orb": np.repeat(1e-4, n_values) * u.Hz, "interpolate_g": False}

        # each of these functions passes whatever it is given on to the sensitivity curve params
        for snr_function in ["get_snr", "get_snr_stationary", "get_snr_evolving"]:
            sources = source.Source(**args)
            self.assertTrue(sources.sc_params["t_obs"] == "auto")

            # the curve should be re-interpolated to match what was supplied
            getattr(sources, snr_function)(t_obs=2 * u.yr, instrument="TianQin")
            self.assertTrue(sources.sc_params["t_obs"] == 2 * u.yr)
            self.assertTrue(sources.sc_params["instrument"] == "TianQin")

            # but supplying the same values again shouldn't repeat the interpolation
            n_interpolations = [0]
            real_set_sc = sources.set_sc

            def counting_set_sc():
                n_interpolations[0] += 1
                real_set_sc()
            sources.set_sc = counting_set_sc

            getattr(sources, snr_function)(t_obs=2 * u.yr, instrument="TianQin")
            self.assertTrue(n_interpolations[0] == 0)

    def test_max_strain_harmonic(self):
        """check that the harmonic with the maximum strain is interpolated sensibly"""
        sources = source.Source(m_1=np.repeat(1, 5) * u.Msun, m_2=np.repeat(1, 5) * u.Msun,
                                f_orb=np.repeat(1e-3, 5) * u.Hz, ecc=np.repeat(0.2, 5),
                                dist=np.repeat(10, 5) * u.kpc, interpolate_g=False, interpolate_sc=False)

        # circular sources radiate at n=2 and more eccentric sources peak at higher harmonics
        self.assertTrue(sources.max_strain_harmonic(0.0) == 2)
        self.assertTrue(np.all(np.diff(sources.max_strain_harmonic(np.array([0.1, 0.5, 0.9]))) > 0))

    def test_bad_sc_params(self):
        """check that only real sensitivity curve params can be set"""
        sources = source.Source(m_1=1 * u.Msun, m_2=1 * u.Msun, f_orb=1e-3 * u.Hz, ecc=0.0, dist=10 * u.kpc,
                                interpolate_g=False, interpolate_sc=False)

        # a parameter that isn't used for the sensitivity curve should be rejected
        for bad_params in [{"not_a_param": 42}, {"instrument": "LISA", "t_ob": 4 * u.yr}]:
            it_broke = False
            try:
                sources.sc_params = bad_params
            except KeyError:
                it_broke = True
            self.assertTrue(it_broke)

        it_broke = False
        try:
            sources.sc_params["nonsense"] = 42
        except KeyError:
            it_broke = True
        self.assertTrue(it_broke)

        # params can be changed but not removed
        it_broke = False
        try:
            del sources.sc_params["instrument"]
        except TypeError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_updating_gw_lum_tol(self):
        """check that changing the GW luminosity tolerance updates the cached calculations"""
        sources = source.Source(m_1=np.repeat(1, 5) * u.Msun, m_2=np.repeat(1, 5) * u.Msun,
                                f_orb=np.repeat(1e-3, 5) * u.Hz, ecc=np.repeat(0.2, 5),
                                dist=np.repeat(10, 5) * u.kpc, interpolate_g=False, interpolate_sc=False)

        original_ecc_tol = sources.ecc_tol
        original_harmonics = sources.harmonics_required(0.3)

        # a tighter tolerance means more harmonics are needed and eccentricity matters sooner
        sources.gw_lum_tol = 0.001

        self.assertTrue(sources.gw_lum_tol == 0.001)
        self.assertTrue(sources.ecc_tol < original_ecc_tol)
        self.assertTrue(sources.harmonics_required(0.3) > original_harmonics)

        # setting the same tolerance again shouldn't repeat the calculations
        unchanged = sources.harmonics_required
        sources.gw_lum_tol = 0.001
        self.assertTrue(sources.harmonics_required is unchanged)

        # `ecc_tol` is derived from the tolerance so it can't be set directly
        it_broke = False
        try:
            sources.ecc_tol = 0.5
        except AttributeError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_semi_major_axis(self):
        """check that assigning to the semi-major axis updates the orbital frequency"""
        n_values = 10
        m_1 = np.random.uniform(1, 10, n_values) * u.Msun
        m_2 = np.random.uniform(1, 10, n_values) * u.Msun
        sources = source.Source(m_1=m_1, m_2=m_2, ecc=np.zeros(n_values),
                                dist=np.repeat(10, n_values) * u.kpc,
                                f_orb=10**(np.random.uniform(-5, -3, n_values)) * u.Hz,
                                interpolate_g=False, interpolate_sc=False)

        new_a = np.logspace(-3, -1, n_values) * u.AU
        sources.a = new_a

        self.assertTrue(np.allclose(sources.a.to(u.AU), new_a))
        self.assertTrue(np.allclose(sources.f_orb, utils.get_f_orb_from_a(new_a, m_1, m_2)))

        # units are still required
        it_broke = False
        try:
            sources.a = np.ones(n_values)
        except AssertionError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_interpolate_g_property(self):
        """check that the class reports whether g(n,e) is interpolated"""
        args = {"m_1": np.repeat(1, 5) * u.Msun, "m_2": np.repeat(1, 5) * u.Msun,
                "f_orb": np.repeat(1e-3, 5) * u.Hz, "ecc": np.zeros(5),
                "dist": np.repeat(10, 5) * u.kpc, "interpolate_sc": False}

        self.assertFalse(source.Source(interpolate_g=False, **args).interpolate_g)
        self.assertTrue(source.Source(interpolate_g=True, **args).interpolate_g)

    @staticmethod
    def _random_sources(n_values=20, positions=False, **kwargs):
        """create a random set of circular sources for testing masking and file IO"""
        m_1 = np.random.uniform(1, 10, n_values) * u.Msun
        m_2 = np.random.uniform(1, 10, n_values) * u.Msun
        dist = np.random.uniform(1, 30, n_values) * u.kpc
        f_orb = 10**(np.random.uniform(-5, -3, n_values)) * u.Hz
        ecc = np.zeros(n_values)

        position = None
        if positions:
            position = SkyCoord(np.random.uniform(0, 360, n_values) * u.deg,
                                np.arcsin(np.random.uniform(-1, 1, n_values)) * u.rad,
                                distance=dist, frame="galactic")

        return source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb, position=position,
                             weights=np.random.uniform(0, 1, n_values), interpolate_g=False, **kwargs)

    def test_masking_sources(self):
        """check that sources can be masked with any index identifier"""
        n_values = 20
        sources = self._random_sources(n_values=n_values)
        sources.get_snr()
        sources.get_merger_time()

        # every type of index should give the same sources as masking the arrays directly
        inds = np.arange(n_values)
        for ind in [5, -1, [1, 3, 5], np.array([0, 7]), slice(2, 8, 2), slice(None),
                    sources.f_orb > 1e-4 * u.Hz]:
            masked = sources[ind]
            expected = inds[[ind]] if isinstance(ind, int) else inds[ind]

            self.assertTrue(len(masked) == len(expected))
            self.assertTrue(masked.n_sources == len(expected))
            self.assertTrue(np.all(masked.m_1 == sources.m_1[expected]))
            self.assertTrue(np.all(masked.m_c == sources.m_c[expected]))
            self.assertTrue(np.all(masked.a == sources.a[expected]))
            self.assertTrue(np.all(masked.ecc == sources.ecc[expected]))
            self.assertTrue(np.all(masked.snr == sources.snr[expected]))
            self.assertTrue(np.all(masked.t_merge == sources.t_merge[expected]))
            self.assertTrue(np.all(masked.weights == sources.weights[expected]))

            # the interpolated functions should be passed straight through
            self.assertTrue(masked.g is sources.g)
            self.assertTrue(masked.sc is sources.sc)

        # the masked class should be able to recompute the same SNRs
        mask = sources.m_1 > 5 * u.Msun
        self.assertTrue(np.allclose(sources[mask].get_snr(), sources.snr[mask]))

        # changing the sc params of the mask shouldn't affect the original class
        masked = sources[mask]
        masked.sc_params["instrument"] = "TianQin"
        self.assertTrue(sources.sc_params["instrument"] == "LISA")

    def test_masking_positions(self):
        """check that positions, inclinations and polarisations are masked too"""
        sources = self._random_sources(positions=True)
        mask = np.random.choice([True, False], sources.n_sources)

        masked = sources[mask]
        self.assertTrue(np.all(masked.position.lon == sources.position.lon[mask]))
        self.assertTrue(np.all(masked.position.lat == sources.position.lat[mask]))
        self.assertTrue(np.all(masked.inclination == sources.inclination[mask]))
        self.assertTrue(np.all(masked.polarisation == sources.polarisation[mask]))

    def test_bad_masks(self):
        """check that nonsense index identifiers are rejected"""
        sources = self._random_sources(n_values=10)

        for bad_ind in [np.repeat(True, 5), 1.5, ["not", "an", "index"]]:
            it_broke = False
            try:
                sources[bad_ind]
            except ValueError:
                it_broke = True
            self.assertTrue(it_broke)

    def test_source_file_io(self):
        """check that sources can be saved to a file and read back in identically"""
        sources = self._random_sources(positions=True, interpolate_sc=True,
                                       sc_params={"t_obs": 5 * u.yr, "confusion_noise": None})
        sources.get_snr()
        sources.get_merger_time()

        with tempfile.TemporaryDirectory() as directory:
            file_name = os.path.join(directory, "sources")
            sources.save(file_name)

            # saving again should only work when overwriting is allowed
            it_broke = False
            try:
                sources.save(file_name)
            except FileExistsError:
                it_broke = True
            self.assertTrue(it_broke)
            sources.save(file_name, overwrite=True)

            loaded = source.Source.from_file(file_name)

            for var in ["m_1", "m_2", "m_c", "dist", "f_orb", "a", "ecc", "weights", "snr", "t_merge",
                        "max_snr_harmonic", "merged", "inclination", "polarisation"]:
                self.assertTrue(np.all(getattr(loaded, var) == getattr(sources, var)))
            self.assertTrue(np.all(loaded.position.lon == sources.position.lon))
            self.assertTrue(np.all(loaded.position.lat == sources.position.lat))

            # settings should match too (including the None confusion noise)
            self.assertTrue(loaded._sc_params == sources._sc_params)
            self.assertTrue(loaded._gw_lum_tol == sources._gw_lum_tol)
            self.assertTrue(loaded.stat_tol == sources.stat_tol)
            self.assertTrue(loaded.interpolate_sc == sources.interpolate_sc)
            self.assertTrue((loaded.g is None) == (sources.g is None))

            # and the loaded sources should give the same SNRs
            self.assertTrue(np.allclose(loaded.get_snr(), sources.snr))

    def test_source_file_io_subclasses(self):
        """check that file IO handles subclasses and interpolation settings"""
        with tempfile.TemporaryDirectory() as directory:
            file_name = os.path.join(directory, "stationary_sources.h5")

            sources = self._random_sources(n_values=10)
            stationary = source.Stationary(m_1=sources.m_1, m_2=sources.m_2, ecc=sources.ecc,
                                           dist=sources.dist, f_orb=sources.f_orb, interpolate_g=False,
                                           interpolate_sc=False)
            stationary.save(file_name)

            # the subclass should be recreated automatically
            loaded = source.Source.from_file(file_name)
            self.assertTrue(isinstance(loaded, source.Stationary))
            self.assertTrue(loaded.g is None and loaded.sc is None)

            # but the user can override the interpolation settings
            loaded = source.Source.from_file(file_name, interpolate_sc=True)
            self.assertTrue(loaded.sc is not None)

            # missing files should be flagged
            it_broke = False
            try:
                source.Source.from_file(os.path.join(directory, "not_a_file.h5"))
            except FileNotFoundError:
                it_broke = True
            self.assertTrue(it_broke)

    def test_reprs(self):
        """check that the reprs are informative and don't crash"""
        sources = self._random_sources(n_values=10)
        repr(sources)
        repr(sources[0])
        repr(sources[0:5])

        self.assertTrue(len(sources) == 10)
        self.assertTrue(len(sources[0:5]) == 5)

        # each subclass should say which type of sources it holds
        args = {"m_1": sources.m_1, "m_2": sources.m_2, "ecc": sources.ecc, "dist": sources.dist,
                "f_orb": sources.f_orb, "interpolate_g": False, "interpolate_sc": False}
        self.assertTrue(repr(source.Stationary(**args)) == "<Stationary: 10 stationary sources>")
        self.assertTrue(repr(source.Evolving(**args)) == "<Evolving: 10 evolving sources>")
        self.assertTrue(repr(source.VerificationBinaries()) == "<VerificationBinaries | Kupfer+2018>")

    def test_source_file_io_warnings(self):
        """check that saving and loading warns the user about anything that can't be reproduced"""
        with tempfile.TemporaryDirectory() as directory:
            file_name = os.path.join(directory, "custom_sources.h5")

            # custom PSD functions can't be written to file
            sources = self._random_sources(n_values=10, interpolate_sc=False,
                                           sc_params={"instrument": "custom", "custom_psd": psd.lisa_psd})

            saving_output = io.StringIO()
            with redirect_stdout(saving_output):
                sources.save(file_name)
            self.assertTrue("custom_psd" in saving_output.getvalue())

            # so the user should be told to supply it again when they load the sources back in
            loading_output = io.StringIO()
            with redirect_stdout(loading_output):
                loaded = source.Source.from_file(file_name)
            self.assertTrue("custom_psd" in loading_output.getvalue())
            self.assertTrue(loaded.sc_params["custom_psd"] is None)
            self.assertTrue(loaded.sc_params["instrument"] == "custom")

            # a file written by a different version of LEGWORK should be flagged
            with h5py.File(file_name, "a") as file:
                file.attrs["legwork_version"] = "0.0.1"

            version_output = io.StringIO()
            with redirect_stdout(version_output):
                source.Source.from_file(file_name)
            self.assertTrue("0.0.1" in version_output.getvalue())
