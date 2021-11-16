import numpy as np
from legwork import evol, utils
import unittest
from scipy.integrate import odeint
from schwimmbad import MultiPool

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
        e = np.random.uniform(0, 0.95, n_values)

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

    def test_bad_input(self):
        """checks that functions handle bad input well"""
        # missing masses
        no_worries = True
        try:
            evol.check_mass_freq_input(beta=None, m_1=None, m_2=None,
                                       f_orb_i=1e-3 * u.Hz, a_i=None)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing individual masses and no a_i
        no_worries = True
        try:
            evol.check_mass_freq_input(beta=10 * u.m**4 / u.s,
                                       f_orb_i=1e-3 * u.Hz, a_i=None)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing frequency and separation
        no_worries = True
        try:
            evol.check_mass_freq_input(m_1=10 * u.Msun, m_2=10 * u.Msun,
                                       f_orb_i=None, a_i=None)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing masses if frequency
        no_worries = True
        try:
            evol.evol_circ(t_evol=None, n_step=100, timesteps=None,
                           beta=1 * u.m**4 / u.s, m_1=None, m_2=None,
                           f_orb_i=1e-3 * u.Hz, a_i=0.1 * u.AU,
                           output_vars='f_orb')
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        # missing masses if frequency
        no_worries = True
        try:
            evol.evol_ecc(ecc_i=0.0, t_evol=None, n_step=100, timesteps=None,
                          beta=1 * u.m**4 / u.s, m_1=None, m_2=None,
                          f_orb_i=1e-3 * u.Hz, a_i=0.1 * u.AU,
                          output_vars='f_orb')
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

        n_vals = 10000
        f_orb_i = 10**(np.random.uniform(-5, -1, n_vals)) * u.Hz
        ecc_i = np.zeros(n_vals)
        m_1 = np.random.uniform(0, 50, n_vals) * u.Msun
        m_2 = np.random.uniform(0, 50, n_vals) * u.Msun

        # check the function doesn't crash if you don't give the chirp mass
        evol.determine_stationarity(f_orb_i=f_orb_i, t_evol=4 * u.yr, ecc_i=ecc_i, m_1=m_1, m_2=m_2)

        # check that it *does* crash when no masses supplied
        no_worries = True
        try:
            evol.determine_stationarity(f_orb_i=f_orb_i, t_evol=4 * u.yr, ecc_i=ecc_i)
        except ValueError:
            no_worries = False
        self.assertFalse(no_worries)

    def test_t_merge_special_cases(self):
        """checks that t_merge_ecc operates properly with exactly circular
        binaries and also single sources"""

        n_values = 10000
        beta = np.random.uniform(10, 50, n_values) * u.AU**4 / u.Gyr
        a_i = np.random.uniform(0.01, 0.1, n_values) * u.AU

        # ensure you get the same value no matter which function you use
        circ_time = evol.get_t_merge_circ(beta=beta, a_i=a_i).to(u.yr)
        ecc_time = evol.get_t_merge_ecc(beta=beta, a_i=a_i,
                                        ecc_i=np.zeros(len(a_i))).to(u.yr)

        self.assertTrue(np.all(circ_time == ecc_time))

        beta = np.random.uniform(10, 50, 1) * u.AU**4 / u.Gyr
        a_i = np.random.uniform(0.01, 0.1, 1) * u.AU

        for e in [0.0, 0.005, 0.5, 0.95]:
            ecc_i = np.array([e])
            # large_e_tol=0.9 so high ecc approximation triggers (for coverage)
            array_time = evol.get_t_merge_ecc(beta=beta, a_i=a_i,
                                              ecc_i=ecc_i,
                                              large_e_tol=0.9).to(u.yr)
            single_time = evol.get_t_merge_ecc(beta=beta[0], a_i=a_i[0],
                                               ecc_i=ecc_i[0],
                                               large_e_tol=0.9).to(u.yr)
            self.assertTrue(array_time[0] == single_time)

    def test_mandel_fit(self):
        """checks that the Mandel fit to the Peters timescale is the same
        as the exact version"""

        n_values = 1000
        beta = np.random.uniform(10, 50, n_values) * u.AU**4 / u.Gyr
        a_i = np.random.uniform(0.01, 0.1, n_values) * u.AU
        ecc = np.random.uniform(0, 0.95, n_values)

        # ensure you get the same value no matter which function you use
        exact_time = evol.get_t_merge_ecc(beta=beta, a_i=a_i, ecc_i=ecc, exact=True).to(u.yr)
        fit_time = evol.get_t_merge_ecc(beta=beta, a_i=a_i, ecc_i=ecc, exact=False).to(u.yr)

        self.assertTrue(np.allclose(exact_time, fit_time, rtol=0.05))

        beta = np.random.uniform(10, 50, 1) * u.AU**4 / u.Gyr
        a_i = np.random.uniform(0.01, 0.1, 1) * u.AU
        ecc = np.random.uniform(0.02, 0.98, 1)  # make sure it is in the middle

        array_time = evol.get_t_merge_ecc(beta=beta, a_i=a_i,
                                          ecc_i=ecc, exact=False).to(u.yr)
        single_time = evol.get_t_merge_ecc(beta=beta[0], a_i=a_i[0],
                                           ecc_i=ecc[0], exact=False).to(u.yr)
        self.assertTrue(array_time[0] == single_time)

    def test_timestep_creation(self):
        n_values = 10

        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.9, n_values)

        a_i = utils.get_a_from_f_orb(f_orb, m_1, m_2)
        beta = utils.beta(m_1, m_2)

        t_merge = evol.get_t_merge_ecc(ecc, a_i=a_i, beta=beta)
        real_times = np.linspace(0 * u.s, t_merge, 100).T
        created_times = evol.create_timesteps_array(a_i=a_i, beta=beta,
                                                    ecc_i=ecc, n_step=100)

        self.assertTrue(np.allclose(real_times, created_times))

        t_evol = np.repeat(4, len(a_i)) * u.yr
        real_times = np.linspace(0 * u.s, t_evol, 100).T
        created_times = evol.create_timesteps_array(a_i=a_i, beta=beta,
                                                    ecc_i=ecc, t_evol=4 * u.yr,
                                                    n_step=100)

        self.assertTrue(np.allclose(real_times, created_times))

        timesteps = np.linspace(0, 100, 100) * u.s
        created_times = evol.create_timesteps_array(a_i=a_i, beta=beta,
                                                    ecc_i=ecc,
                                                    timesteps=timesteps)
        real_times = timesteps[np.newaxis, :]
        real_times = np.broadcast_to(timesteps.value,
                                     (n_values, 100)) * timesteps.unit
        self.assertTrue(np.allclose(real_times, created_times))

    def test_evol_output_vars(self):

        m_1 = np.random.uniform(0, 10) * u.Msun
        m_2 = np.random.uniform(0, 10) * u.Msun
        f_orb = 10**(np.random.uniform(-5, -1)) * u.Hz
        ecc = np.random.uniform(0.0, 0.9)

        a_i = utils.get_a_from_f_orb(f_orb, m_1, m_2)

        evolution = evol.evol_circ(m_1=m_1, m_2=m_2, a_i=a_i,
                                   output_vars=["a", "f_GW", "timesteps"])
        self.assertTrue(len(evolution) == 3)

        evolution = evol.evol_ecc(ecc_i=ecc, m_1=m_1, m_2=m_2, a_i=a_i,
                                  output_vars=["a", "f_GW", "timesteps"])
        self.assertTrue(len(evolution) == 3)

    def test_de_dt_integrate(self):
        n_values = 10

        m_1 = np.random.uniform(0, 10, n_values) * u.Msun
        m_2 = np.random.uniform(0, 10, n_values) * u.Msun
        f_orb = 10**(np.random.uniform(-5, -1, n_values)) * u.Hz
        ecc = np.random.uniform(0.0, 0.9, n_values)
        beta, a_i = evol.check_mass_freq_input(m_1=m_1, m_2=m_2, f_orb_i=f_orb)
        n_step = 100
        c_0 = utils.c_0(a_i=a_i, ecc_i=ecc)
        timesteps = evol.create_timesteps_array(a_i=a_i, beta=beta, ecc_i=ecc,
                                                t_evol=1 * u.yr, n_step=n_step)

        t_merge = evol.get_t_merge_ecc(ecc_i=ecc, f_orb_i=f_orb, m_1=m_1, m_2=m_2)

        # remove any bad timesteps that would evolve past the merger
        bad_timesteps = timesteps >= t_merge[:, np.newaxis]
        timesteps[bad_timesteps] = -1 * u.Gyr
        previous = timesteps.max(axis=1).repeat(timesteps.shape[1])
        timesteps[bad_timesteps] = previous.reshape(timesteps.shape)[bad_timesteps]

        # get rid of the units for faster integration
        c_0 = c_0.to(u.m).value
        beta = beta.to(u.m**4 / u.s).value
        timesteps = timesteps.to(u.s).value

        # integrate by hand:
        ecc_evol = np.array([odeint(evol.de_dt, ecc[i], timesteps[i],
                                    args=(beta[i], c_0[i])).flatten()
                             for i in range(len(ecc))])

        # integrate with function:
        with MultiPool(processes=1) as pool:
            ecc_pool = np.array(list(pool.map(evol.integrate_de_dt,
                                              zip(ecc,
                                                  timesteps,
                                                  beta,
                                                  c_0))))

        self.assertTrue(np.allclose(ecc_evol, ecc_pool, equal_nan=True))


# need to use the main name for multiprocessing to work
if __name__ == '__main__':
    unittest.main()
