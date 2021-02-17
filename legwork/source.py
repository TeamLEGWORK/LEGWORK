"""A collection of classes for analysing gravitational wave sources"""
from astropy import units as u
import numpy as np
from importlib import resources
from scipy.interpolate import interp1d, interp2d

import legwork.utils as utils
import legwork.strain as strain
import legwork.snr as sn
import legwork.visualisation as vis

__all__ = ['Source', 'Stationary', 'Evolving']


class Source():
    """Class for generic GW sources

    This class is for analysing a generic set of sources that may be
    stationary/evolving and circular/eccentric. If the type of sources are
    known, then a more specific subclass may be more useful

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass. Must have astropy units of mass.

    m_2 : `float/array`
        secondary mass. Must have astropy units of mass.

    ecc : `float/array`
        initial eccentricity

    dist : `float/array`
        luminosity distance to source. Must have astropy units of distance.

    f_orb : `float/array`
        orbital frequency (either `a` or `f_orb` must be supplied)
        This takes precedence over `a`. Must have astropy units of frequency.

    a : `float/array`
        semi-major axis (either `a` or `f_orb` must be supplied). Must have
        astropy units of length.

    gw_lum_tol : `float`
        allowed error on the GW luminosity when calculating snrs.
        This is used to calculate maximum harmonics needed and
        transition between 'eccentric' and 'circular'.
        This variable should be updated using the function
        :meth:`legwork.source.Source.update_gw_lum_tol` (not
        ``Source._gw_lum_tol =``) to ensure the cached calculations match the
        current tolerance.

    stat_tol : `float`
        fractional change in frequency above which a
        binary should be considered to be stationary

    interpolate_g : `boolean`
        whether to interpolate the g(n,e) function from Peters (1964)

    Attributes
    ----------
    m_c : array_like
        chirp mass. Set using `m_1` and `m_2` in
        :meth:`legwork.utils.chirp_mass`

    ecc_tol : float
        eccentricity above which a binary is considered eccentric. Set by
        :meth:`legwork.source.Source.find_eccentric_transition`

    Raises
    ------
    ValueError
        If both `f_orb` and `a` are missing.
        If array-like parameters don't have the same length.

    AssertionError
        If a parameter is missing units
    """
    def __init__(self, m_1, m_2, ecc, dist, f_orb=None, a=None,
                 gw_lum_tol=0.05, stat_tol=1e-2, interpolate_g=True):
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
        """Create two harmonics related functions

        1. to calculate the maximum harmonics required to calculate the SNRs
        assuming provided tolerance `gw_lum_tol`
        
        2. to calculate the dominant harmonic frequency (max strain)"""

        # open file containing pre-calculated g(n,e) and F(e) values
        with resources.path(package="legwork",
                            resource="harmonics.npz") as path:
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
        interpolated_hn = interp1d(e_range, harmonics_needed,
                                   bounds_error=False,
                                   fill_value=(2, np.max(harmonics_needed)))

        # conservatively round up to nearest integer
        def max_harmonic(e):
            return np.ceil(interpolated_hn(e)).astype(int)
        self.max_harmonic = max_harmonic

        # now calculate the dominant harmonics
        dominant_harmonics = n_range[g_vals.argmax(axis=1)]
        interpolated_dh = interp1d(e_range, dominant_harmonics,
                                   bounds_error=False,
                                   fill_value=(2, np.max(harmonics_needed)))

        def dominant_harmonic(e):   # pragma: no cover
            return np.round(interpolated_dh(e)).astype(int)

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

        Parameters
        ----------
        gw_lum_tol : `float`
            allowed error on the GW luminosity when calculating snrs
        """
        self._gw_lum_tol = gw_lum_tol
        self.create_harmonics_functions()
        self.find_eccentric_transition()

    def set_g(self, interpolate_g):
        """Set Source g function if user wants to interpolate g(n,e).
        Otherwise just leave the function as None.

        Parameters
        ----------
        interpolate_g : `boolean`
            whether to interpolate the g(n,e) function from Peters (1964)
        """
        if interpolate_g:
            # open file containing pre-calculated fine g(n,e) grid
            with resources.path(package="legwork",
                                resource="peters_g.npy") as path:
                peters_g = np.load(path)

            # interpolate grid using scipy
            n_range = np.arange(1, 10000 + 1).astype(int)
            e_range = np.linspace(0, 1, 1000)
            self.g = interp2d(n_range, e_range, peters_g, kind="cubic")
        else:
            self.g = None

    def get_source_mask(self, circular=None, stationary=None, t_obs=4 * u.yr):
        """Produce a mask of the sources based on whether binaries
        are circular or eccentric and stationary or evolving.
        Tolerance levels are defined in the class.

        Parameters
        ----------
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
            stat_mask = np.repeat(True, self.n_sources)
        elif stationary is True or stationary is False:
            stat_mask = utils.determine_stationarity(m_c=self.m_c,
                                                     f_orb_i=self.f_orb,
                                                     t_evol=t_obs,
                                                     ecc_i=self.ecc,
                                                     stat_tol=self.stat_tol)
            if stationary is False:
                stat_mask = np.logical_not(stat_mask)
        else:
            raise ValueError("`stationary` must be None, True or False")

        return np.logical_and(circular_mask, stat_mask)

    def get_h_0_n(self, harmonics, which_sources=None):
        """Computes the strain for all binaries for the given `harmonics`

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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
        stat_mask = self.get_source_mask(circular=None, stationary=True,
                                         t_obs=t_obs)
        evol_mask = np.logical_not(stat_mask)

        if stat_mask.any():
            if verbose:
                n_stat = len(snr[stat_mask])
                print("\t{} sources are stationary".format(n_stat))
            snr[stat_mask] = self.get_snr_stationary(t_obs=t_obs,
                                                     which_sources=stat_mask,
                                                     verbose=verbose)
        if evol_mask.any():
            if verbose:
                n_evol = len(snr[evol_mask])
                print("\t{} sources are evolving".format(n_evol))
            snr[evol_mask] = self.get_snr_evolving(t_obs=t_obs,
                                                   which_sources=evol_mask,
                                                   n_step=n_step,
                                                   verbose=verbose)
        self.snr = snr
        return snr

    def get_snr_stationary(self, t_obs=4 * u.yr, which_sources=None,
                           verbose=False):
        """Computes the SNR assuming a stationary binary

        Parameters
        ----------
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

        Parameters
        ----------
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

    def plot_source_variables(self, xstr, ystr=None, which_sources=None,
                              **kwargs):  # pragma: no cover
        """plot distributions of Source variables. If two variables are
        specified then produce a 2D distribution, otherwise a 1D distribution.

        Parameters
        ----------

        xstr : `{ 'm_1', 'm_2', 'm_c', 'ecc', 'dist', 'f_orb', 'f_GW', 'a',\
                'snr' }`
            which variable to plot on the x axis

        ystr : `{ 'm_1', 'm_2', 'm_c', 'ecc', 'dist', 'f_orb', 'f_GW', 'a',\
                snr' }`
            which variable to plot on the y axis
            (if None then a 1D distribution is made using `xstr`)

        which_sources : `boolean array`
            mask for which sources should be plotted (default is all sources)

        **kwargs : `various`
            When only `xstr` is provided, the kwargs are the same as
            :meth:`legwork.visualisation.plot_1D_dist`. When both `xstr` and
            `ystr` are provided, the kwargs are the same as
            :meth:`legwork.visualisation.plot_2D_dist`. Note that if `xlabel`
            or `ylabel` is not passed then this function automatically creates
            one using a default string and (if applicable) the Astropy units
            of the variable.

        Returns
        -------
        fig : `matplotlib Figure`
            the figure on which the distribution is plotted

        ax : `matplotlib Axis`
            the axis on which the distribution is plotted
        """
        convert = {"m_1": self.m_1, "m_2": self.m_2,
                   "m_c": self.m_c,
                   "ecc": self.ecc * u.dimensionless_unscaled,
                   "dist": self.dist, "f_orb": self.f_orb,
                   "f_GW": self.f_orb * 2, "a": self.a,
                   "snr": self.snr * u.dimensionless_unscaled
                   if self.snr is not None else self.snr}
        labels = {"m_1": "Primary Mass", "m_2": "Secondary Mass",
                  "m_c": "Chirp Mass", "ecc": "Eccentricity",
                  "dist": "Distance", "f_orb": "Orbital Frequency",
                  "f_GW": "Gravitational Wave Frequency",
                  "a": "Semi-major axis", "snr": "Signal-to-noise Ratio"}
        unitless = set(["ecc", "snr"])

        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

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
            return vis.plot_2D_dist(x=x[which_sources].value,
                                    y=y[which_sources].value, **kwargs)
        else:
            return vis.plot_1D_dist(x=x[which_sources].value, **kwargs)

    def plot_sources_on_sc(self, snr_cutoff=0, t_obs=4 * u.yr, fig=None,
                           ax=None, show=True, **kwargs):  # pragma: no cover
        """plot all sources in the class on the sensitivity curve

        Parameters
        ----------
        snr_cutoff : `float`
            SNR below which sources will not be plotted (default is to plot
            all sources)

        t_obs : `float`
            LISA observation time

        show : `boolean`
            whether to immediately show the plot

        Returns
        -------
        fig : `matplotlib Figure`
            the figure on which the sources are plotted

        ax : `matplotlib Axis`
            the axis on which the sources are plotted
        """
        # plot circular and stationary sources
        circ_stat = self.get_source_mask(circular=True, stationary=True)
        if circ_stat.any():
            f_orb = self.f_orb[circ_stat]
            h_0_2 = self.get_h_0_n(2, which_sources=circ_stat).flatten()
            fig, ax = vis.plot_sources_on_sc_circ_stat(f_orb=f_orb,
                                                       h_0_2=h_0_2,
                                                       snr=self.snr[circ_stat],
                                                       snr_cutoff=snr_cutoff,
                                                       t_obs=t_obs,
                                                       fig=fig, ax=ax,
                                                       show=False,
                                                       **kwargs)

        # plot eccentric and stationary sources
        ecc_stat = self.get_source_mask(circular=False, stationary=True)
        if ecc_stat.any():
            n_dom = self.dominant_harmonic(self.ecc[ecc_stat])
            f_dom = self.f_orb[ecc_stat] * n_dom
            fig, ax = vis.plot_sources_on_sc_ecc_stat(f_dom=f_dom,
                                                      snr=self.snr[ecc_stat],
                                                      snr_cutoff=snr_cutoff,
                                                      t_obs=t_obs, show=show,
                                                      fig=fig, ax=ax, **kwargs)

        # show warnings for evolving sources
        circ_evol = self.get_source_mask(circular=True, stationary=False)
        if circ_evol.any():
            print("{} circular and evolving".format(len(circ_evol[circ_evol])),
                  "sources detected, plotting not yet implemented for",
                  "evolving sources.")

        ecc_evol = self.get_source_mask(circular=False, stationary=False)
        if ecc_evol.any():
            print("{} eccentric and evolving".format(len(ecc_evol[ecc_evol])),
                  "sources detected, plotting not yet implemented for",
                  "evolving sources.")

        return fig, ax


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
