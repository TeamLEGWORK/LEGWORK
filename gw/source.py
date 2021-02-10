"""`circular binary stuff!`"""
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from importlib import resources
from scipy.interpolate import interp1d, interp2d

import gw.utils as utils
import gw.strain as strain
import gw.snr as sn
import gw.visualisation as vis

__all__ = ['Source', 'Stationary', 'Evolving']


class Source():
    """Superclass for generic sources"""
    def __init__(self, m_1, m_2, ecc, dist, f_orb=None, a=None,
                 gw_lum_tol=0.05, stat_tol=1e-2, interpolate_g=True):
        """Initialise all parameters
        Params
        ------
        m_1 : `float/array`
            primary mass

        m_2 : `float/array`
            secondary mass

        ecc : `float/array`
            initial eccentricity

        dist : `float/array`
            luminosity distance to source

        forb : `float/array`
            orbital frequency (either a or forb must be supplied)
            This takes precedence over a

        a : `float/array`
            semi-major axis (either a or forb must be supplied)

        gw_lum_tol : `float`
            allowed error on the GW luminosity when calculating snrs.
            This is used to calculate maximum harmonics needed and
            transition between 'eccentric' and 'circular'.
            This variable should be updated using the function
            `update_gw_lum_tol` (not Source._gw_lum_tol =) to ensure
            the cached calculations match the current tolerance.

        stat_tol : `float`
            fractional change in frequency above which a
            binary should be considered to be stationary


        interpolate_g : `boolean`
            whether to interpolate the g(n,e) function from Peters (1964)
        """
        # ensure that either a frequency or semi-major axis is supplied
        if f_orb is None and a is None:
            raise ValueError("Either `f_orb` or `a` must be specified")

        # calculate whichever one wasn't supplied
        f_orb = utils.get_f_orb_from_a(a, m_1, m_2) if f_orb is None else f_orb
        a = utils.get_a_from_f_orb(f_orb, m_1, m_2) if a is None else a

        # define which arguments must have units
        unit_args_str = ['m_1', 'm_2', 'dist', 'f_orb', 'a']

        # define which arguments must be arrays of same length
        array_args = [m_1, m_2, dist, f_orb, a, ecc]
        array_args_str = ['m_1', 'm_2', 'dist', 'f_orb', 'a', 'ecc']

        # convert args to numpy arrays if only single values are entered
        for i in range(len(array_args)):
            if array_args_str[i] in unit_args_str:
                # check that every arg has units if it should
                assert(isinstance(array_args[i], u.quantity.Quantity)), \
                        "`{}` must have units".format(array_args_str[i])

                if not isinstance(array_args[i].value, (np.ndarray, list)):
                    array_args[i] = np.array([array_args[i].value]) \
                                  * array_args[i].unit
            else:
                if not isinstance(array_args[i], (np.ndarray, list)):
                    array_args[i] = np.array([array_args[i]])

        # ensure all array arguments are the same length
        length_check = np.array([len(arg) != len(array_args[0])
                                 for arg in array_args])
        if length_check.any():
            raise ValueError("All input arrays must have the same length")

        # reset the arguments with the new converted ones
        m_1, m_2, dist, f_orb, a, ecc = array_args

        self.m_1 = m_1
        self.m_2 = m_2
        self.m_c = utils.chirp_mass(m_1, m_2)
        self.ecc = ecc
        self.dist = dist
        self.stat_tol = stat_tol
        self.f_orb = f_orb
        self.a = a
        self.snr = None
        self.n_sources = len(m_1)

        self.update_gw_lum_tol(gw_lum_tol)
        self.set_g(interpolate_g)

    def create_harmonics_functions(self):
        """Create two functions:
        1. to calculate the maximum harmonics required to calculate the SNRs
        assuming provided tolerance `gw_lum_tol`
        2. to calculate the dominant harmonic frequency (max strain)"""

        # open file containing pre-calculated g(n,e) and F(e) values
        with resources.path(package="gw", resource="harmonics.npz") as path:
            lum_info = np.load(path)

        e_min, e_max, e_len = lum_info["e_lims"]
        e_len = e_len.astype(int)
        n_max = lum_info["n_max"]
        g_vals = lum_info["g_vals"]

        # reconstruct arrays
        e_range = 1 - np.logspace(np.log10(1 - e_min),
                                  np.log10(1 - e_max), e_len)
        n_range = np.arange(1, n_max.astype(int) + 1)

        f_vals = utils.peters_f(e_range)

        # set harmonics needed to 2 for a truly circular system (base case)
        harmonics_needed = np.zeros(e_len).astype(int)
        harmonics_needed[0] = 2

        for i in range(1, e_len):
            # harmonics needed are at least as many as lower eccentricity
            harmonics_needed[i] = harmonics_needed[i - 1]
            total_lum = g_vals[i][:harmonics_needed[i]].sum()

            # keep adding harmonics until gw luminosity is within errors
            while total_lum < (1 - self._gw_lum_tol) * f_vals[i] \
                    and harmonics_needed[i] < len(n_range):
                harmonics_needed[i] += 1
                total_lum += g_vals[i][harmonics_needed[i] - 1]

        # interpolate the answer and return the max if e > e_max
        interpolated = interp1d(e_range, harmonics_needed, bounds_error=False,
                                fill_value=(2, np.max(harmonics_needed)))

        # conservatively round up to nearest integer
        def max_harmonic(e):
            return np.ceil(interpolated(e)).astype(int)
        self.max_harmonic = max_harmonic

        # now calculate the dominant harmonics
        dominant_harmonics = n_range[g_vals.argmax(axis=1)]
        interpolated = interp1d(e_range, dominant_harmonics, bounds_error=False,
                                fill_value=(2, np.max(harmonics_needed)))
        def dominant_harmonic(e):
            return np.round(interpolated(e)).astype(int)
        self.dominant_harmonic = dominant_harmonic

    def find_eccentric_transition(self):
        """Find the eccentricity at which we must treat binaries at eccentric.
        We define this as the maximum eccentricity at which the n=2 harmonic
        is the total GW luminosity given the tolerance `self._gw_lum_tol`"""
        # only need to check lower eccentricities
        e_range = np.linspace(0.0, 0.2, 10000)

        # find first e where n=2 harmonic is below tolerance
        circular_lum = utils.peters_g(2, e_range)
        lum_within_tolerance = (1 - self._gw_lum_tol) * utils.peters_f(e_range)
        self.ecc_tol = e_range[circular_lum < lum_within_tolerance][0]

    def update_gw_lum_tol(self, gw_lum_tol):
        """Update GW luminosity tolerance and use updated value to
        recalculate max_harmonics function and transition to eccentric

        Params
        ------
        gw_lum_tol : `float`
            allowed error on the GW luminosity when calculating snrs
        """
        self._gw_lum_tol = gw_lum_tol
        self.create_harmonics_functions()
        self.find_eccentric_transition()

    def set_g(self, interpolate_g):
        """Set Source g function if user wants to interpolate g(n,e).
        Otherwise just leave the function as None.

        Params
        ------
        interpolate_g : `boolean`
            whether to interpolate the g(n,e) function from Peters (1964)
        """
        if interpolate_g:
            # open file containing pre-calculated fine g(n,e) grid
            with resources.path(package="gw",
                                resource="peters_g.npy") as path:
                peters_g = np.load(path)

            # interpolate grid using scipy
            n_range = np.arange(1, 10000 + 1).astype(int)
            e_range = np.linspace(0, 1, 1000)
            self.g = interp2d(n_range, e_range, peters_g)
        else:
            self.g = None

    def get_source_mask(self, circular=None, stationary=None, t_obs=4 * u.yr):
        """Produce a mask of the sources based on whether binaries
        are circular or eccentric and stationary or evolving.
        Tolerance levels are defined in the class.

        Params
        ------
        circular : `bool`
            `None` means either, `True` means only circular
            binaries and `False` means only eccentric

        stationary : `bool`
            `None` means either, `True` means only stationary
            binaries and `False` means only evolving

        t_obs : `float`
            observation time

        Returns
        -------
        mask : `bool/array`
            mask for the sources
        """
        if circular is None:
            circular_mask = np.repeat(True, self.n_sources)
        elif circular is True:
            circular_mask = self.ecc <= self.ecc_tol
        elif circular is False:
            circular_mask = self.ecc > self.ecc_tol
        else:
            raise ValueError("`circular` must be None, True or False")

        if stationary is None:
            stationary_mask = np.repeat(True, self.n_sources)
        elif stationary is True or stationary is False:
            stationary_mask = utils.determine_stationarity(m_c=self.m_c,
                                                           f_orb_i=self.f_orb,
                                                           t_evol=t_obs,
                                                           ecc_i=self.ecc,
                                                           stat_tol=self.stat_tol)
            if stationary is False:
                stationary_mask = np.logical_not(stationary_mask)
        else:
            raise ValueError("`stationary` must be None, True or False")

        return np.logical_and(circular_mask, stationary_mask)

    def get_h_0_n(self, harmonics, which_sources=None):
        """Computes the strain for all binaries for the given `harmonics`

        Params
        ------
        harmonics : `int/array`
            harmonic(s) at which to calculate the strain

        which_sources : `boolean/array`
            mask on which sources to compute values for (default is all)

        Returns
        -------
        h_0_n : `float/array`
            dimensionless strain in the quadrupole approximation (unitless)
            shape of array is `(number of sources, number of harmonics)`
        """
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        return strain.h_0_n(m_c=self.m_c[which_sources],
                            f_orb=self.f_orb[which_sources],
                            ecc=self.ecc[which_sources],
                            n=harmonics,
                            dist=self.dist[which_sources],
                            interpolated_g=self.g)

    def get_h_c_n(self, harmonics, which_sources=None):
        """Computes the characteristic strain for all binaries
        for the given `harmonics`

        Params
        ------
        harmonics : `int/array`
            harmonic(s) at which to calculate the strain

        which_sources `boolean/array`
            mask on which sources to compute values for (default is all)

        Returns
        -------
        h_c_n : `float/array`
            dimensionless characteristic strain in the quadrupole approximation
            shape of array is `(number of sources, number of harmonics)`
        """
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        return strain.h_c_n(m_c=self.m_c[which_sources],
                            f_orb=self.f_orb[which_sources],
                            ecc=self.ecc[which_sources],
                            n=harmonics,
                            dist=self.dist[which_sources],
                            interpolated_g=self.g)

    def get_snr(self, t_obs=4 * u.yr, n_step=100, verbose=False):
        """Computes the SNR for a generic binary

        Params
        ------
        t_obs : `array`
            observation duration (default: 4 years)

        n_step : `int`
            number of time steps during observation duration

        verbose : `boolean`
            whether to print additional information to user

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        if verbose:
            print("Calculating SNR for {} sources".format(self.n_sources))
        snr = np.zeros(self.n_sources)
        stationary_mask = self.get_source_mask(circular=None, stationary=True,
                                               t_obs=t_obs)
        evolving_mask = np.logical_not(stationary_mask)

        if stationary_mask.any():
            if verbose:
                n_stat = len(snr[stationary_mask])
                print("\t{} sources are stationary".format(n_stat))
            snr[stationary_mask] = self.get_snr_stationary(t_obs=t_obs,
                                                           which_sources=stationary_mask,
                                                           verbose=verbose)
        if evolving_mask.any():
            if verbose:
                n_evol = len(snr[evolving_mask])
                print("\t{} sources are evolving".format(n_evol))
            snr[evolving_mask] = self.get_snr_evolving(t_obs=t_obs,
                                                       which_sources=evolving_mask,
                                                       n_step=n_step,
                                                       verbose=verbose)
        self.snr = snr
        return snr

    def get_snr_stationary(self, t_obs=4 * u.yr, which_sources=None,
                           verbose=False):
        """Computes the SNR assuming a stationary binary

        Params
        ------
        t_obs : `array`
            observation duration in units of yr

        which_sources : `bool/array`
            mask on which sources to consider stationary and calculate
            (default is all sources in Class)

        verbose : `boolean`
            whether to print additional information to user

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        snr = np.zeros(self.n_sources)
        ind_ecc = np.logical_and(self.ecc > self.ecc_tol, which_sources)
        ind_circ = np.logical_and(self.ecc <= self.ecc_tol, which_sources)

        # only compute snr if there is at least one binary in mask
        if ind_circ.any():
            if verbose:
                print("\t\t{} sources are stationary and circular".format(
                    len(snr[ind_circ])))
            snr[ind_circ] = sn.snr_circ_stationary(m_c=self.m_c[ind_circ],
                                                   f_orb=self.f_orb[ind_circ],
                                                   dist=self.dist[ind_circ],
                                                   t_obs=t_obs,
                                                   interpolated_g=self.g)
        if ind_ecc.any():
            if verbose:
                print("\t\t{} sources are stationary and eccentric".format(
                    len(snr[ind_ecc])))
            max_harmonics = self.max_harmonic(self.ecc)
            harmonic_groups = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
            for lower, upper in harmonic_groups:
                harm_mask = np.logical_and(max_harmonics >= lower,
                                           max_harmonics < upper)
                match = np.logical_and(harm_mask, ind_ecc)
                if match.any():
                    snr[match] = sn.snr_ecc_stationary(m_c=self.m_c[match],
                                                       f_orb=self.f_orb[match],
                                                       ecc=self.ecc[match],
                                                       dist=self.dist[match],
                                                       t_obs=t_obs,
                                                       max_harmonic=upper - 1,
                                                       interpolated_g=self.g)

        return snr[which_sources]

    def get_snr_evolving(self, t_obs, n_step=100, which_sources=None,
                         verbose=False):
        """Computes the SNR assuming an evolving binary

        Params
        ------
        t_obs : `array`
            observation duration (default: 4 years)

        n_step : `int`
            number of time steps during observation duration

        which_sources : `bool/array`
            mask on which sources to consider evolving and calculate
            (default is all sources in Class)

        verbose : `boolean`
            whether to print additional information to user

        Returns
        -------
        SNR : `array`
            the signal to noise ratio
        """
        snr = np.zeros(self.n_sources)

        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        ind_ecc = np.logical_and(self.ecc > self.ecc_tol, which_sources)
        ind_circ = np.logical_and(self.ecc <= self.ecc_tol, which_sources)

        if ind_circ.any():
            if verbose:
                print("\t\t{} sources are evolving and circular".format(
                    len(snr[ind_circ])))
            snr[ind_circ] = sn.snr_circ_evolving(m_1=self.m_1[ind_circ],
                                                 m_2=self.m_2[ind_circ],
                                                 f_orb_i=self.f_orb[ind_circ],
                                                 dist=self.dist[ind_circ],
                                                 t_obs=t_obs,
                                                 n_step=n_step,
                                                 interpolated_g=self.g)
        if ind_ecc.any():
            if verbose:
                print("\t\t{} sources are evolving and eccentric".format(
                    len(snr[ind_ecc])))
            max_harmonics = self.max_harmonic(self.ecc)
            harmonic_groups = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
            for lower, upper in harmonic_groups:
                harm_mask = np.logical_and(max_harmonics >= lower,
                                           max_harmonics < upper)
                match = np.logical_and(harm_mask, ind_ecc)
                if match.any():
                    snr[match] = sn.snr_ecc_evolving(m_1=self.m_1[match],
                                                     m_2=self.m_2[match],
                                                     f_orb_i=self.f_orb[match],
                                                     dist=self.dist[match],
                                                     ecc=self.ecc[match],
                                                     max_harmonic=upper - 1,
                                                     t_obs=t_obs,
                                                     n_step=n_step,
                                                     interpolated_g=self.g)

        return snr[which_sources]

    def plot_source_variables(self, xstr, ystr=None, **kwargs):
        """plot distributions of Source variables. If two variables are
        specified then produce a 2D distribution, otherwise a 1D distribution.
        This function is a wrapper on `visualisation.plot_1D_dist` and
        `visualisation.plot_2D_dist` and thus takes mostly the same parameters.

        Params
        ------
        xstr : `{{ 'm_1', 'm_2', 'm_c', 'ecc', 'dist', 'f_orb', 'f_GW', 'a',
        snr' }}`
            which variable to plot on the x axis

        ystr : `{{ 'm_1', 'm_2', 'm_c', 'ecc', 'dist', 'f_orb', 'f_GW', 'a',
        snr' }}`
            which variable to plot on the y axis
            (if None then a 1D distribution is made using `xstr`)

        Keyword Args
        ------------
        These are exactly the same as `visualisation.plot_1D_dist`, see those
        docs for more details.

        Note that if `xlabel` or `ylabel` is not passed then this function
        automatically creates one using a default string and (if applicable)
        the units of the variable.

        Returns
        -------
        fig : `matplotlib Figure`
            the figure on which the distribution is plotted

        ax : `matplotlib Axis`
            the axis on which the distribution is plotted
        """
        convert = {"m_1": self.m_1, "m_2": self.m_2,
                   "m_c": utils.chirp_mass(self.m_1, self.m_2),
                   "ecc": self.ecc * u.dimensionless_unscaled,
                   "dist": self.dist, "f_orb": self.f_orb,
                   "f_GW": self.f_orb * 2, "a": self.a,
                   "snr": self.snr * u.dimensionless_unscaled}
        labels = {"m_1": "Primary Mass", "m_2": "Secondary Mass",
                  "m_c": "Chirp Mass", "ecc": "Eccentricity",
                  "dist": "Distance", "f_orb": "Orbital Frequency",
                  "f_GW": "Gravitational Wave Frequency",
                  "a": "Semi-major axis", "snr": "Signal-to-noise Ratio"}
        unitless = set(["ecc", "snr"])

        # ensure that the variable is a valid choice
        for var_str in [xstr, ystr]:
            if var_str not in convert.keys() and var_str is not None:
                error_str = "`xstr` and `ystr` must be one of: " \
                    + ', '.join(["`{}`".format(k)
                                 for k in list(convert.keys())])
                raise ValueError(error_str)

        # check the instance variable has been already set
        x = convert[xstr]
        if x is None:
            raise ValueError("x variable (`{}`)".format(xstr),
                             "must be not be None")
        if ystr is not None:
            y = convert[ystr]
            if y is None:
                raise ValueError("y variable (`{}`)".format(ystr),
                                 "must be not be None")

        # create the x label if it wasn't provided
        if "xlabel" not in kwargs.keys():
            if xstr in unitless:
                kwargs["xlabel"] = labels[xstr]
            else:
                kwargs["xlabel"] = r"{} [{:latex}]".format(labels[xstr],
                                                           x.unit)

        # create the y label if it wasn't provided and ystr was
        if ystr is not None and "ylabel" not in kwargs.keys():
            if ystr in unitless:
                kwargs["ylabel"] = labels[ystr]
            else:
                kwargs["ylabel"] = r"{} [{:latex}]".format(labels[ystr],
                                                           y.unit)

        # plot it!
        if ystr is not None:
            return vis.plot_2D_dist(x=x.value, y=y.value, **kwargs)
        else:
            return vis.plot_1D_dist(x=x.value, **kwargs)


class Stationary(Source):
    """Subclass for sources that are stationary"""

    def get_snr(self, t_obs=4*u.yr, verbose=False):
        self.snr = self.get_snr_stationary(t_obs=t_obs, verbose=verbose)
        return self.snr


class Evolving(Source):
    """Subclass for sources that are evolving"""

    def get_snr(self, t_obs=4*u.yr, n_step=100, verbose=False):
        self.snr = self.get_snr_evolving(t_obs=t_obs, n_step=n_step,
                                         verbose=verbose)
        return self.snr
