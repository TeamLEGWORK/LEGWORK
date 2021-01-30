"""`circular binary stuff!`"""
from astropy import units as u
import numpy as np
from scipy import interpolate

import calcs.utils as utils
import calcs.snr as sn
from importlib import resources
from scipy.interpolate import interp1d

__all__ = ['Source', 'Stationary', 'Evolving']

class Source():
    """Superclass for generic sources"""
    def __init__(self, m_1, m_2, f_orb, ecc, dist, gw_lum_tol=0.05, stat_tol=1e-2):
        """Initialise all parameters
        Params
        ------
        m_1 : `float/array`
            primary mass

        m_2 : `float/array`
            secondary mass

        forb : `float/array`
            orbital frequency

        ecc : `float/array`
            initial eccentricity

        dist : `float/array`
            luminosity distance to source

        gw_lum_tol : `float`
            allowed error on the GW luminosity when calculating snrs.
            This is used to calculate maximum harmonics needed and
            transition between 'eccentric' and 'circular'.
            This variable should be updated using the function
            `update_gw_lum_tol` (not Source.gw_lum_tol =) to ensure
            the cached calculations match the current tolerance.

        stat_tol : `float`
            fractional change in frequency above which a
            binary should be considered to be stationary
        """
        self.m_1 = m_1
        self.m_2 = m_2
        self.f_orb = f_orb
        self.ecc = ecc
        self.dist = dist
        self.stat_tol = stat_tol
        self.n_sources = len(m_1)
        self.snr = None

        self.update_gw_lum_tol(gw_lum_tol)

    def create_max_harmonics_function(self):
        """Create a function to calculate the maximum harmonics required
        to calculate the SNRs assuming provided tolerance `gw_lum_tol`"""

        # open file containing pre-calculating g(n,e) and F(e) values
        with resources.path(package="calcs", resource="harmonics.npz") as path:
            lum_info = np.load(path)

        e_min, e_max, e_len = lum_info["e_lims"]
        e_len = e_len.astype(int)
        n_max = lum_info["n_max"]
        g_vals = lum_info["g_vals"]
            
        # reconstruct arrays
        e_range = 1 - np.logspace(np.log10(1 - e_min), np.log10(1 - e_max), e_len)
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
            while total_lum < (1 - self.gw_lum_tol) * f_vals[i] \
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

    def find_eccentric_transition(self):
        """Find the eccentricity at which we must treat binaries at eccentric.
        We define this as the maximum eccentricity at which the n=2 harmonic
        is the total GW luminosity given the tolerance `self.gw_lum_tol`"""
        # only need to check lower eccentricities
        e_range = np.linspace(0.0, 0.2, 10000)

        # find first e where n=2 harmonic is below tolerance
        circular_lum = utils.peters_g(2, e_range)
        lum_within_tolerance = (1 - self.gw_lum_tol) * utils.peters_f(e_range)
        self.ecc_tol = e_range[circular_lum < lum_within_tolerance][0]

    def update_gw_lum_tol(self, gw_lum_tol):
        """Update GW luminosity tolerance and use updated value to
        recalculate max_harmonics function and transition to eccentric
        
        Params
        ------
        gw_lum_tol : `float`
            allowed error on the GW luminosity when calculating snrs
        """
        self.gw_lum_tol = gw_lum_tol
        self.create_max_harmonics_function()
        self.find_eccentric_transition()

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
        elif stationary is True:
            stationary_mask = utils.determine_stationarity(self.m_1, self.m_2,
                                                           self.f_orb, t_obs,
                                                           self.ecc, self.stat_tol)
        elif stationary is False:
            stationary_mask = np.logical_not(utils.determine_stationarity(self.m_1, self.m_2,
                                                           self.f_orb, t_obs,
                                                           self.ecc, self.stat_tol))
        else:
            raise ValueError("`stationary` must be None, True or False")

        return np.logical_and(circular_mask, stationary_mask)

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
        stationary_mask = self.get_source_mask(circular=None, stationary=True, t_obs=t_obs)
        evolving_mask = np.logical_not(stationary_mask)

        if stationary_mask.any():
            if verbose:
                print("\t{} sources are stationary".format(len(snr[stationary_mask])))
            snr[stationary_mask] = self.get_snr_stationary(t_obs=t_obs,
                                                           which_sources=stationary_mask,
                                                           verbose=verbose)
        if evolving_mask.any():
            if verbose:
                print("\t{} sources are evolving".format(len(snr[evolving_mask])))
            snr[evolving_mask] = self.get_snr_evolving(t_obs=t_obs,
                                                       which_sources=evolving_mask,
                                                       n_step=n_step,
                                                       verbose=verbose)
        self.snr = snr
        return snr

    def get_snr_stationary(self, t_obs=4 * u.yr, which_sources=None, verbose=False):
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
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)

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
            snr[ind_circ] = sn.snr_circ_stationary(m_c=m_c[ind_circ], 
                                                   f_orb=self.f_orb[ind_circ], 
                                                   dist=self.dist[ind_circ], 
                                                   t_obs=t_obs)
        if ind_ecc.any():
            if verbose:
                print("\t\t{} sources are stationary and eccentric".format(
                    len(snr[ind_ecc])))
            max_harmonics = self.max_harmonic(self.ecc)
            harmonic_groups = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
            for lower, upper in harmonic_groups:
                matching = np.logical_and(np.logical_and(max_harmonics >= lower, max_harmonics < upper), ind_ecc)
                if matching.any():
                    snr[matching] = sn.snr_ecc_stationary(m_c=m_c[matching],
                                                        f_orb=self.f_orb[matching],
                                                        ecc=self.ecc[matching],
                                                        dist=self.dist[matching],
                                                        t_obs=t_obs,
                                                        max_harmonic=upper - 1)

        return snr[which_sources]

    def get_snr_evolving(self, t_obs, n_step=100, which_sources=None, verbose=False):
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
        m_c = utils.chirp_mass(m_1=self.m_1, m_2=self.m_2)
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
                                                 n_step=n_step)
        if ind_ecc.any():
            if verbose:
                print("\t\t{} sources are evolving and eccentric".format(
                    len(snr[ind_ecc])))
            max_harmonic = np.max(self.max_harmonic(self.ecc[ind_ecc]))
            snr[ind_ecc] = sn.snr_ecc_evolving(m_1=self.m_1[ind_ecc],
                                               m_2=self.m_2[ind_ecc],
                                               f_orb_i=self.f_orb[ind_ecc],
                                               dist=self.dist[ind_ecc],
                                               ecc=self.ecc[ind_ecc],
                                               max_harmonic=max_harmonic,
                                               t_obs=t_obs,
                                               n_step=n_step)

        return snr[which_sources]

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
