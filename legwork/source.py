"""A collection of classes for analysing gravitational wave sources"""
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from importlib import resources
from scipy.interpolate import interp1d, interp2d

from legwork import utils, strain, psd, evol
import legwork.snr as sn
import legwork.visualisation as vis

__all__ = ['Source', 'Stationary', 'Evolving', 'VerificationBinaries']


class Source():
    """Class for generic GW sources

    This class is for analysing a generic set of sources that may be stationary/evolving and
    circular/eccentric. If the type of sources are known, then a more specific subclass may be more useful

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass. Must have astropy units of mass.

    m_2 : `float/array`
        Secondary mass. Must have astropy units of mass.

    ecc : `float/array`
        Initial eccentricity

    dist : `float/array`
        Luminosity distance to source. Must have astropy units of distance.

    n_proc : `int`
        Number of processors to split eccentric evolution over if needed

    f_orb : `float/array`
        Orbital frequency (either `a` or `f_orb` must be supplied). This takes precedence over `a`.
        Must have astropy units of frequency.

    a : `float/array`
        Semi-major axis (either `a` or `f_orb` must be supplied). Must have astropy units of length.

    position : `SkyCoord/array`, optional
        Sky position of source. Must be specified using Astropy's :class:`astropy.coordinates.SkyCoord` class.

    polarisation : `float/array`, optional
        GW polarisation angle of the source. Must have astropy angular units.

    inclination : `float/array`, optional
        Inclination of the source. Must have astropy angular units.

    weights : `float/array`, optional
        Statistical weights associated with each sample (used for plotted), default is equal weights

    gw_lum_tol : `float`
        Allowed error on the GW luminosity when calculating SNRs. This is used to calculate maximum harmonics
        needed and transition between 'eccentric' and 'circular'. This variable should be updated using the
        function :meth:`legwork.source.Source.update_gw_lum_tol` (not ``Source._gw_lum_tol =``) to ensure
        the cached calculations match the current tolerance.

    stat_tol : `float`
        Fractional change in frequency over mission length above which a binary should be considered to be
        stationary

    interpolate_g : `boolean` or 'auto'
        Whether to interpolate the g(n,e) function from Peters (1964). If
        'auto' is inputted then LEGWORK will decide whether it is
        necessary to interpolate g(n,e) based on the number of sources and
        their eccentricity.

    interpolate_sc : `boolean`
        Whether to interpolate the LISA sensitivity curve

    sc_params : `dict`
        Parameters for interpolated sensitivity curve. Include any of ``instrument``, ``custom_psd``,
        ``t_obs``, ``L``, ``approximate_R`` and ``confusion_noise``. Default values are: "LISA", None,
        4 years, 2.5e9, 19.09e-3, False and 'robson19'.

    Attributes
    ----------
    m_c : `float/array`
        Chirp mass. Set using ``m_1`` and ``m_2`` in :meth:`legwork.utils.chirp_mass`

    ecc_tol : `float`
        Eccentricity above which a binary is considered eccentric. Set by
        :meth:`legwork.source.Source.find_eccentric_transition`

    snr : `float/array`
        Signal-to-noise ratio. Set by :meth:`legwork.source.Source.get_snr`

    max_snr_harmonic : `int/array`
        Harmonic with the maximum snr. Set by :meth:`legwork.source.Source.get_snr`

    n_sources : `int`
        Number of sources in class

    Raises
    ------
    ValueError
        If both ``f_orb`` and ``a`` are missing.
        If only part of the position, inclination, and polarization are supplied.
        If array-like parameters don't have the same length.

    AssertionError
        If a parameter is missing units
    """

    def __init__(self, m_1, m_2, ecc, dist, n_proc=1, f_orb=None, a=None, position=None, polarisation=None,
                 inclination=None, weights=None, gw_lum_tol=0.05, stat_tol=1e-2, interpolate_g="auto",
                 interpolate_sc=True, sc_params={}):
        # ensure that either a frequency or semi-major axis is supplied
        if f_orb is None and a is None:
            raise ValueError("Either `f_orb` or `a` must be specified")

        # raise errors or fill missing values
        if position is None:
            if inclination is not None:
                raise ValueError("If you specify the inclination, you must also specify a sky position.")

            if polarisation is not None:
                raise ValueError("If you specify the polarisation, you must also specify a sky position.")
        else:
            if inclination is None:
                print("Generating random values for source inclinations")
                inclination = np.arcsin(np.random.uniform(-1, 1, len(m_1))) * u.rad
            if polarisation is None:
                print("Generating random values for source polarisations")
                polarisation = np.random.uniform(0, 2 * np.pi, len(m_1)) * u.rad

        # ensure position is in the correct coordinate frame
        if position is not None:
            if np.atleast_1d(ecc).any() > 0.0:
                raise ValueError("The sky position, inclination, and polarization "
                                 "modulation is only valued for circular sources")

            # ensure position is in the correct coordinate frame
            position = position.transform_to("heliocentrictrueecliptic")

            # ensure that the position, polarisation, and inclination
            # quantities are at least 1d for masking later on
            lon, lat, polarisation, inclination = np.atleast_1d(position.lon, position.lat,
                                                                polarisation, inclination)
            position = SkyCoord(lon=lon, lat=lat, distance=dist, frame='heliocentrictrueecliptic')

        # calculate whichever one wasn't supplied
        f_orb = utils.get_f_orb_from_a(a, m_1, m_2) if f_orb is None else f_orb
        a = utils.get_a_from_f_orb(f_orb, m_1, m_2) if a is None else a

        # define which arguments must have units
        unit_args = [m_1, m_2, dist, f_orb, a]
        unit_args_str = ['m_1', 'm_2', 'dist', 'f_orb', 'a']

        for i in range(len(unit_args)):
            assert (isinstance(unit_args[i], u.quantity.Quantity)), \
                "`{}` must have units".format(unit_args_str[i])

        # make sure the inputs are arrays
        fixed_args, _ = utils.ensure_array(m_1, m_2, dist, f_orb, a, ecc, weights)
        m_1, m_2, dist, f_orb, a, ecc, weights = fixed_args

        # ensure all array arguments are the same length
        array_args = [m_1, m_2, dist, f_orb, a, ecc]
        length_check = np.array([len(arg) != len(array_args[0])
                                 for arg in array_args])
        if length_check.any():
            raise ValueError("All input arrays must have the same length")

        default_sc_params = {
            "instrument": "LISA",
            "custom_psd": None,
            "t_obs": 4 * u.yr,
            "L": 2.5e9 * u.m,
            "approximate_R": False,
            "confusion_noise": 'robson19'
        }
        default_sc_params.update(sc_params)
        self._sc_params = default_sc_params

        self.m_1 = m_1
        self.m_2 = m_2
        self.m_c = utils.chirp_mass(m_1, m_2)
        self.ecc = ecc
        self.dist = dist
        self.stat_tol = stat_tol
        self.f_orb = f_orb
        self.a = a
        self.position = position
        self.inclination = inclination
        self.polarisation = polarisation
        self.weights = weights
        self.n_proc = n_proc
        self.t_merge = None
        self.snr = None
        self.max_snr_harmonic = None
        self.n_sources = len(m_1)
        self.interpolate_sc = interpolate_sc

        self.merged = np.repeat(False, self.n_sources)

        self.update_gw_lum_tol(gw_lum_tol)

        # interpolate g(n,e) for more than 100 sources or eccentric populations
        if interpolate_g == "auto":
            self.set_g(np.logical_or(self.n_sources > 100, np.any(self.ecc > 0.9)))
        else:
            self.set_g(interpolate_g)
        self.set_sc()

    def create_harmonics_functions(self):
        """Create two harmonics related functions as methods for the Source class

        The first function is stored at ``self.harmonics_required(ecc)``. This calculates the index of the
        highest harmonic required to calculate the SNR of a system with eccentricity `ecc` assuming the
        provided tolerance `gw_lum_tol`. This is equivalent to the total number of harmonics required since,
        when calculating SNR, harmonics in the range [1, harmonics_required(ecc)] are used. Note that the
        value returned by the function slightly conservative as we apply `ceil` to the interpolation result.

        The second function is stored at ``self.max_strain_harmonic(ecc)``. This calculates the harmonic with
        the maximum strain for a system with eccentricity `ecc`."""

        # open file containing pre-calculated g(n,e) and F(e) values
        with resources.path(package="legwork", resource="harmonics.npz") as path:
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
            while total_lum < (1 - self._gw_lum_tol) * f_vals[i] and harmonics_needed[i] < len(n_range):
                harmonics_needed[i] += 1
                total_lum += g_vals[i][harmonics_needed[i] - 1]

        # interpolate the answer and return the max if e > e_max
        interpolated_hn = interp1d(e_range, harmonics_needed, bounds_error=False,
                                   fill_value=(2, np.max(harmonics_needed)))

        # conservatively round up to nearest integer
        def harmonics_required(e):
            return np.ceil(interpolated_hn(e)).astype(int)

        self.harmonics_required = harmonics_required

        # now calculate the max strain harmonics
        max_strain_harmonics = n_range[g_vals.argmax(axis=1)]
        interpolated_dh = interp1d(e_range, max_strain_harmonics, bounds_error=False,
                                   fill_value=(2, np.max(harmonics_needed)))

        def max_strain_harmonic(e):  # pragma: no cover
            return np.round(interpolated_dh(e)).astype(int)

        self.max_strain_harmonic = max_strain_harmonic

    def find_eccentric_transition(self):
        """Find the eccentricity at which we must treat binaries at eccentric. We define this as the maximum
        eccentricity at which the n=2 harmonic is the total GW luminosity given the tolerance
        ``self._gw_lum_tol``. Store the result in ``self.ecc_tol``"""
        # only need to check lower eccentricities
        e_range = np.linspace(0.0, 0.2, 10000)

        # find first e where n=2 harmonic is below tolerance
        circular_lum = utils.peters_g(2, e_range)
        lum_within_tolerance = (1 - self._gw_lum_tol) * utils.peters_f(e_range)
        self.ecc_tol = e_range[circular_lum < lum_within_tolerance][0]

    def update_gw_lum_tol(self, gw_lum_tol):
        """Update GW luminosity tolerance. Use the updated value to recalculate harmonics_required function
        and transition to eccentric

        Parameters
        ----------
        gw_lum_tol : `float`
            Allowed error on the GW luminosity when calculating SNRs
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
            Whether to interpolate the g(n,e) function from Peters (1964)
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

    def set_sc(self):
        """Set Source sensitivity curve function

        If user wants to interpolate then perform interpolation of LISA sensitivity curve using
        ``sc_params``. Otherwise just leave the function as None.
        """
        if self.interpolate_sc:
            # get values
            frequency_range = np.logspace(-7, np.log10(2), 10000) * u.Hz
            sc = psd.power_spectral_density(frequency_range, **self._sc_params)

            # interpolate
            interp_sc = interp1d(frequency_range, sc, bounds_error=False, fill_value=1e30)

            # add units back
            self.sc = lambda f: interp_sc(f.to(u.Hz)) / u.Hz
        else:
            self.sc = None

    def update_sc_params(self, sc_params):
        """Update sensitivity curve parameters

        Update the parameters used to interpolate sensitivity curve and perform interpolation again to
        match new params
        """
        # check whether params have actually changed
        if sc_params != self._sc_params:
            # ensure all values are filled (leave as defaults if not)
            default_sc_params = {
                "instrument": "LISA",
                "custom_psd": None,
                "t_obs": 4 * u.yr,
                "L": 2.5e9 * u.m,
                "approximate_R": False,
                "confusion_noise": "robson19"
            }
            if sc_params is not None:
                default_sc_params.update(sc_params)
            # change values and re-interpolate
            self._sc_params = default_sc_params
            self.set_sc()

    def get_source_mask(self, circular=None, stationary=None, t_obs=4 * u.yr):
        """Produce a mask of the sources.

        Create a mask based on whether binaries are circular or eccentric and stationary or evolving.
        Tolerance levels are defined in the class.

        Parameters
        ----------
        circular : `bool`
            ``None`` means either, ``True`` means only circular binaries and ``False`` means only eccentric

        stationary : `bool`
            ``None`` means either, ``True`` means only stationary binaries and ``False`` means only evolving

        t_obs : `float`
            Observation time

        Returns
        -------
        mask : `bool/array`
            Mask for the sources
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
            stat_mask = evol.determine_stationarity(m_c=self.m_c, f_orb_i=self.f_orb, t_evol=t_obs,
                                                    ecc_i=self.ecc, stat_tol=self.stat_tol)
            if stationary is False:
                stat_mask = np.logical_not(stat_mask)
        else:
            raise ValueError("`stationary` must be None, True or False")

        return np.logical_and(circular_mask, stat_mask)

    def get_h_0_n(self, harmonics, which_sources=None):
        """Computes the strain for binaries for the given ``harmonics``. Use ``which_sources`` to select a
        subset of the sources. Merged sources are set to have 0.0 strain.

        Parameters
        ----------
        harmonics : `int/array`
            Harmonic(s) at which to calculate the strain

        which_sources : `boolean/array`
            Mask on which sources to compute values for (default is all)

        Returns
        -------
        h_0_n : `float/array`
            Dimensionless strain in the quadrupole approximation (unitless) shape of array is
            ``(number of sources, number of harmonics)``
        """
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

        # by default set all strains to zero
        n_harmonics = len(harmonics) if not isinstance(harmonics, int) else 1
        h_0_n = np.zeros((self.n_sources, n_harmonics))

        # find a mask for the inspiraling sources (exclude merged)
        insp_sources = np.logical_and(np.logical_not(self.merged), which_sources)

        # only apply the mask to the position, polarization and inclination
        # if they are provided
        position = self.position[insp_sources] if self.position is not None else None
        polarisation = self.polarisation[insp_sources] if self.position is not None else None
        inclination = self.inclination[insp_sources] if self.position is not None else None

        # calculate strain for these values
        h_0_n[insp_sources, :] = strain.h_0_n(m_c=self.m_c[insp_sources],
                                              f_orb=self.f_orb[insp_sources],
                                              ecc=self.ecc[insp_sources],
                                              n=harmonics,
                                              dist=self.dist[insp_sources],
                                              position=position,
                                              polarisation=polarisation,
                                              inclination=inclination,
                                              interpolated_g=self.g)[:, 0, :]

        # return all sources, not just inspiraling ones
        return h_0_n[which_sources, :]

    def get_h_c_n(self, harmonics, which_sources=None):
        """Computes the characteristic strain for binaries for the given ``harmonics``. Use
        ``which_sources`` to select a subset of the sources. Merged sources are set to have 0.0
        characteristic strain.

        Parameters
        ----------
        harmonics : `int/array`
            Harmonic(s) at which to calculate the strain

        which_sources `boolean/array`
            Mask on which sources to compute values for (default is all)

        Returns
        -------
        h_c_n : `float/array`
            Dimensionless characteristic strain in the quadrupole approximation shape of array is
            ``(number of sources, number of harmonics)``
        """
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

        # by default set all strains to zero
        n_harmonics = len(harmonics) if not isinstance(harmonics, int) else 1
        h_c_n = np.zeros((self.n_sources, n_harmonics))

        # find a mask for the inspiralling sources (exclude merged)
        insp_sources = np.logical_and(np.logical_not(self.merged), which_sources)

        # only apply the mask to the position, polarization and inclination
        # if they are provided
        position = self.position[insp_sources] if self.position is not None else None
        polarisation = self.polarisation[insp_sources] if self.position is not None else None
        inclination = self.inclination[insp_sources] if self.position is not None else None

        # calculate strain for these values
        h_c_n[insp_sources, :] = strain.h_c_n(m_c=self.m_c[insp_sources],
                                              f_orb=self.f_orb[insp_sources],
                                              ecc=self.ecc[insp_sources],
                                              n=harmonics,
                                              dist=self.dist[insp_sources],
                                              position=position,
                                              polarisation=polarisation,
                                              inclination=inclination,
                                              interpolated_g=self.g)[:, 0, :]

        # return all sources, not just inpsiralling ones
        return h_c_n[which_sources, :]

    def get_snr(self, t_obs=None, instrument=None, custom_psd=None, n_step=100,
                verbose=False, re_interpolate_sc=True, which_sources=None):
        """Computes the SNR for a generic binary. Also records the harmonic with maximum SNR for each
        binary in ``self.max_snr_harmonic``.

        Parameters
        ----------
        t_obs : `array`
            Observation duration (default: value from sc_params)

        instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
            Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied. (default: value
            from sc_params)

        custom_psd : `function`
            Custom function for computing the PSD. Must take the same arguments as
            :meth:`legwork.psd.lisa_psd` even if it ignores some. (default: function from sc_params)

        n_step : `int`
            Number of time steps during observation duration

        verbose : `boolean`
            Whether to print additional information to user

        re_interpolate_sc : `boolean`
            Whether to re-interpolate the sensitivity curve if the observation time or instrument
            changes. If False, warning will instead be given

        which_sources : `boolean/array`
            Mask of which sources to calculate the SNR for. If None then calculate SNR for all sources.

        Returns
        -------
        SNR : `array`
            The signal-to-noise ratio
        """
        # if no values are provided, use those in sc_params
        t_obs = self._sc_params["t_obs"] if t_obs is None else t_obs
        instrument = self._sc_params["instrument"] if instrument is None else instrument
        custom_psd = self._sc_params["custom_psd"] if custom_psd is None else custom_psd

        # if the user interpolated a sensitivity curve with different settings
        if (self.interpolate_sc and self._sc_params is not None
                and (t_obs != self._sc_params["t_obs"]
                     or instrument != self._sc_params["instrument"]
                     or custom_psd != self._sc_params["custom_psd"])):  # pragma: no cover

            # re interpolate the sensitivity curve with new parameters
            if re_interpolate_sc:
                self._sc_params["t_obs"] = t_obs
                self._sc_params["instrument"] = instrument
                self._sc_params["custom_psd"] = custom_psd

                self.set_sc()

            # otherwise warn the user that they are making a mistake
            else:
                print("WARNING: Current `sc_params` are different from what was passed to this function.",
                      "Either set `re_interpolate_sc=True` to re-interpolate the sensitivity curve on the",
                      "fly or update your `sc_params` with Source.update_sc_params() to make sure your",
                      "interpolated curve matches")

        if verbose:
            n_snr = len(which_sources[which_sources]) if which_sources is not None else self.n_sources
            print("Calculating SNR for {} sources".format(n_snr))
            print("\t{}".format(len(self.merged[self.merged])),
                  "sources have already merged")
        snr = np.zeros(self.n_sources)

        # by default calculate SNR for every source
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

        stat_mask = np.logical_and.reduce((self.get_source_mask(circular=None,
                                                                stationary=True,
                                                                t_obs=t_obs),
                                           np.logical_not(self.merged),
                                           which_sources))
        evol_mask = np.logical_and.reduce((self.get_source_mask(circular=None,
                                                                stationary=False,
                                                                t_obs=t_obs),
                                           np.logical_not(self.merged),
                                           which_sources))

        if stat_mask.any():
            if verbose:
                n_stat = len(snr[stat_mask])
                print("\t{} sources are stationary".format(n_stat))
            snr[stat_mask] = self.get_snr_stationary(t_obs=t_obs,
                                                     instrument=instrument,
                                                     custom_psd=custom_psd,
                                                     which_sources=stat_mask,
                                                     verbose=verbose)
        if evol_mask.any():
            if verbose:
                n_evol = len(snr[evol_mask])
                print("\t{} sources are evolving".format(n_evol))
            snr[evol_mask] = self.get_snr_evolving(t_obs=t_obs,
                                                   instrument=instrument,
                                                   custom_psd=custom_psd,
                                                   which_sources=evol_mask,
                                                   n_step=n_step,
                                                   verbose=verbose)
        return snr

    def get_snr_stationary(self, t_obs=4 * u.yr, instrument="LISA", custom_psd=None, which_sources=None,
                           verbose=False):
        """Computes the SNR assuming a stationary binary

        Parameters
        ----------
        t_obs : `array`
            Observation duration (default: 4 years)

        instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
            Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied.

        custom_psd : `function`
            Custom function for computing the PSD. Must take the same arguments as
            :meth:`legwork.psd.lisa_psd` even if it ignores some.

        which_sources : `bool/array`
            Mask on which sources to consider stationary and calculate (default is all sources in Class)

        verbose : `boolean`
            Whether to print additional information to user

        Returns
        -------
        SNR : `array`
            The signal-to-noise ratio
        """
        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

        insp_sources = np.logical_and(which_sources, np.logical_not(self.merged))
        snr = np.zeros(self.n_sources)
        e_mask = np.logical_and(self.ecc > self.ecc_tol, insp_sources)
        c_mask = np.logical_and(self.ecc <= self.ecc_tol, insp_sources)

        # default to n = 2 for max snr harmonic
        msh = np.repeat(2, self.n_sources)

        # only apply the mask to the position, polarization and inclination
        # if they are provided
        position = self.position[c_mask] if self.position is not None else None
        polarisation = self.polarisation[c_mask] if self.position is not None else None
        inclination = self.inclination[c_mask] if self.position is not None else None

        # only compute snr if there is at least one binary in mask
        if c_mask.any():
            if verbose:
                print("\t\t{} sources are stationary and circular".format(len(snr[c_mask])))
            snr[c_mask] = sn.snr_circ_stationary(m_c=self.m_c[c_mask],
                                                 f_orb=self.f_orb[c_mask],
                                                 dist=self.dist[c_mask],
                                                 t_obs=t_obs,
                                                 interpolated_g=self.g,
                                                 interpolated_sc=self.sc,
                                                 instrument=instrument,
                                                 custom_psd=custom_psd,
                                                 position=position,
                                                 polarisation=polarisation,
                                                 inclination=inclination)
        if e_mask.any():
            if verbose:
                print("\t\t{} sources are stationary and eccentric".format(len(snr[e_mask])))
            harmonics_required = self.harmonics_required(self.ecc)
            harmonic_groups = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
            for lower, upper in harmonic_groups:
                harm_mask = np.logical_and(harmonics_required > lower, harmonics_required <= upper)
                match = np.logical_and(harm_mask, e_mask)

                # only apply the mask to the position, polarization and inclination
                # if they are provided
                position = self.position[match] if self.position is not None else None
                polarisation = self.polarisation[match] if self.position is not None else None
                inclination = self.inclination[match] if self.position is not None else None

                if match.any():
                    snr[match], msh[match] = sn.snr_ecc_stationary(m_c=self.m_c[match],
                                                                   f_orb=self.f_orb[match],
                                                                   ecc=self.ecc[match],
                                                                   dist=self.dist[match],
                                                                   t_obs=t_obs,
                                                                   harmonics_required=upper,
                                                                   interpolated_g=self.g,
                                                                   interpolated_sc=self.sc,
                                                                   ret_max_snr_harmonic=True,
                                                                   instrument=instrument,
                                                                   custom_psd=custom_psd,
                                                                   position=position,
                                                                   polarisation=polarisation,
                                                                   inclination=inclination)

        if self.max_snr_harmonic is None:
            self.max_snr_harmonic = np.zeros(self.n_sources).astype(int)
        self.max_snr_harmonic[insp_sources] = msh[insp_sources]

        if self.snr is None:
            self.snr = np.zeros(self.n_sources)
        self.snr[insp_sources] = snr[insp_sources]

        return snr[which_sources]

    def get_snr_evolving(self, t_obs, instrument="LISA", custom_psd=None, n_step=100, which_sources=None,
                         verbose=False):
        """Computes the SNR assuming an evolving binary

        Parameters
        ----------
        t_obs : `array`
            Observation duration (default: 4 years)

        instrument : `{{ 'LISA', 'TianQin', 'custom' }}`
            Instrument to observe with. If 'custom' then ``custom_psd`` must be supplied.

        custom_psd : `function`
            Custom function for computing the PSD. Must take the same arguments as
            :meth:`legwork.psd.lisa_psd` even if it ignores some.

        n_step : `int`
            Number of time steps during observation duration

        which_sources : `bool/array`
            Mask on which sources to consider evolving and calculate (default is all sources in Class)

        verbose : `boolean`
            Whether to print additional information to user

        Returns
        -------
        SNR : `array`
            The signal-to-noise ratio
        """
        snr = np.zeros(self.n_sources)

        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

        insp_sources = np.logical_and(which_sources, np.logical_not(self.merged))
        e_mask = np.logical_and(self.ecc > self.ecc_tol, insp_sources)
        c_mask = np.logical_and(self.ecc <= self.ecc_tol, insp_sources)

        # only apply the mask to the position, polarization and inclination
        # if they are provided
        position = self.position[c_mask] if self.position is not None else None
        polarisation = self.polarisation[c_mask] if self.position is not None else None
        inclination = self.inclination[c_mask] if self.position is not None else None

        # default to n = 2 for max snr harmonic
        msh = np.repeat(2, self.n_sources)

        if c_mask.any():
            if verbose:
                print("\t\t{} sources are evolving and circular".format(len(snr[c_mask])))
            t_merge = None if self.t_merge is None else self.t_merge[c_mask]
            snr[c_mask] = sn.snr_circ_evolving(m_1=self.m_1[c_mask],
                                               m_2=self.m_2[c_mask],
                                               f_orb_i=self.f_orb[c_mask],
                                               dist=self.dist[c_mask],
                                               t_obs=t_obs,
                                               n_step=n_step,
                                               t_merge=t_merge,
                                               interpolated_g=self.g,
                                               interpolated_sc=self.sc,
                                               instrument=instrument,
                                               custom_psd=custom_psd,
                                               position=position,
                                               polarisation=polarisation,
                                               inclination=inclination)
        if e_mask.any():
            if verbose:
                print("\t\t{} sources are evolving and eccentric".format(len(snr[e_mask])))
            harmonics_required = self.harmonics_required(self.ecc)
            harmonic_groups = [(1, 10), (10, 100), (100, 1000), (1000, 10000)]
            for lower, upper in harmonic_groups:
                harm_mask = np.logical_and(harmonics_required > lower, harmonics_required <= upper)
                match = np.logical_and(harm_mask, e_mask)
                # only apply the mask to the position, polarization and inclination
                # if they are provided
                position = self.position[match] if self.position is not None else None
                polarisation = self.polarisation[match] if self.position is not None else None
                inclination = self.inclination[match] if self.position is not None else None

                if match.any():
                    t_merge = None if self.t_merge is None else self.t_merge[match]
                    snr[match], msh[match] = sn.snr_ecc_evolving(m_1=self.m_1[match],
                                                                 m_2=self.m_2[match],
                                                                 f_orb_i=self.f_orb[match],
                                                                 dist=self.dist[match],
                                                                 ecc=self.ecc[match],
                                                                 harmonics_required=upper,
                                                                 t_obs=t_obs,
                                                                 n_step=n_step,
                                                                 interpolated_g=self.g,
                                                                 interpolated_sc=self.sc,
                                                                 n_proc=self.n_proc,
                                                                 ret_max_snr_harmonic=True,
                                                                 instrument=instrument,
                                                                 custom_psd=custom_psd,
                                                                 position=position,
                                                                 polarisation=polarisation,
                                                                 inclination=inclination)

        if self.max_snr_harmonic is None:
            self.max_snr_harmonic = np.zeros(self.n_sources).astype(int)
        self.max_snr_harmonic[insp_sources] = msh[insp_sources]

        if self.snr is None:
            self.snr = np.zeros(self.n_sources)
        self.snr[insp_sources] = snr[insp_sources]

        return snr[which_sources]

    def get_merger_time(self, save_in_class=True, which_sources=None, exact=True):
        """Get the merger time for each source. Set ``save_in_class`` to true to save the values as an
        instance variable in the class. Use ``which_sources`` to select a subset of the sources in the
        class. Note that if ``save_in_class`` is set to ``True``, ``which_sources`` will be ignored.

        Parameters
        ----------
        save_in_class : `bool`, optional
            Whether the save the result into the class as an instance variable, by default True
        which_sources : `bool/array`, optional
            A mask for the subset of sources for which to calculate the merger time,
            by default all sources (None)
        exact : `boolean`, optional
            Whether to calculate the merger time exactly with numerical
            integration or to instead use a fit

        Returns
        -------
        t_merge : `float/array`
            Merger times
        """
        # if no subset or saving times, select all sources
        if save_in_class or which_sources is None:
            which_sources = np.repeat(True, self.n_sources)
        t_merge = np.zeros(len(which_sources)) * u.Gyr

        # only compute merger times for inspiralling binaries
        insp = np.logical_and(which_sources, np.logical_not(self.merged))
        t_merge[insp] = evol.get_t_merge_ecc(ecc_i=self.ecc[insp], f_orb_i=self.f_orb[insp],
                                             m_1=self.m_1[insp], m_2=self.m_2[insp], exact=exact)
        if save_in_class:
            self.t_merge = t_merge

        # only return subset of sources
        return t_merge[which_sources]

    def evolve_sources(self, t_evol, create_new_class=False):
        """Evolve sources forward in time for ``t_evol`` amount of time. If ``create_new_class`` is
        ``True`` then save the updated sources in a new Source class, otherwise, update the values
        in this class.

        Parameters
        ----------
        t_evol : `float/array`
            Amount of time to evolve sources. Either a single value for all sources or an array of values
            corresponding to each source.

        create_new_class : bool, optional
            Whether to save the evolved binaries in a new class or not. If not simply update the current
            class, by default False.

        Returns
        -------
        evolved_sources : `Source`
            The new class with evolved sources, only returned if ``create_new_class`` is ``True``.
        """
        # if merger time hasn't be calculated, calculate a quick fit for it
        if self.t_merge is None:
            self.get_merger_time()

        merged = t_evol >= self.t_merge

        # separate out the exactly circular sources from eccentric ones
        c_mask = np.logical_and(self.ecc == 0.0, np.logical_not(merged))
        e_mask = np.logical_and(self.ecc != 0.0, np.logical_not(merged))

        # split up the evolution times if need be
        if isinstance(t_evol.value, (int, float)):
            t_evol_circ = t_evol
            t_evol_ecc = t_evol
        else:
            t_evol_circ = t_evol[c_mask]
            t_evol_ecc = t_evol[e_mask]

        # set up the evolved eccentricity and frequency arrays
        n_step = 2
        ecc_evol = np.zeros(self.n_sources)
        f_orb_evol = np.zeros(self.n_sources) * u.Hz

        # set the frequency of the merged objects
        f_orb_evol[merged] = 100 * u.Hz

        # calculate the evolved values for circular binaries
        if c_mask.any():
            evolution = evol.evol_circ(t_evol=t_evol_circ, n_step=n_step, m_1=self.m_1[c_mask],
                                       m_2=self.m_2[c_mask], f_orb_i=self.f_orb[c_mask], output_vars="f_orb")
            f_orb_evol[c_mask] = evolution[:, -1]

        # calculate the evolved values for eccentric binaries
        if e_mask.any():
            evolution = evol.evol_ecc(t_evol=t_evol_ecc, n_step=n_step, m_1=self.m_1[e_mask],
                                      m_2=self.m_2[e_mask], f_orb_i=self.f_orb[e_mask],
                                      ecc_i=self.ecc[e_mask], output_vars=["ecc", "f_orb"],
                                      avoid_merger=False)

            # drop everything except the final evolved value
            ecc_evol[e_mask] = evolution[0][:, -1]
            f_orb_evol[e_mask] = evolution[1][:, -1]

        if create_new_class:
            # create new source with same attributes (but evolved ecc/f_orb)
            evolved_sources = Source(m_1=self.m_1, m_2=self.m_2, ecc=ecc_evol, dist=self.dist,
                                     n_proc=self.n_proc, f_orb=f_orb_evol, position=self.position,
                                     polarisation=self.polarisation, inclination=self.inclination,
                                     gw_lum_tol=self._gw_lum_tol, stat_tol=self.stat_tol,
                                     interpolate_g=False, interpolate_sc=False, sc_params=self._sc_params)

            # copy over interpolated g and sc
            evolved_sources.g = self.g
            evolved_sources.interpolate_sc = True
            evolved_sources.sc = self.sc
            evolved_sources.t_merge = None

            if self.t_merge is not None:
                evolved_sources.t_merge = np.maximum(0 * u.Gyr, self.t_merge - t_evol)

            # record which sources have merged
            evolved_sources.merged = merged

            # return the new source class
            return evolved_sources
        else:
            # otherwise just update the existing class with evolution
            self.ecc = ecc_evol
            self.f_orb = f_orb_evol
            self.merged = merged

            if self.t_merge is not None:
                self.t_merge = np.maximum(0 * u.Gyr, self.t_merge - t_evol)

    def plot_source_variables(self, xstr, ystr=None, which_sources=None,
                              exclude_merged_sources=True, **kwargs):  # pragma: no cover
        """Plot distributions of Source variables. If two variables are specified then produce a 2D
        distribution, otherwise a 1D distribution.

        Parameters
        ----------
        xstr : `{ 'm_1', 'm_2', 'm_c', 'ecc', 'dist', 'f_orb', 'f_GW', 'a', 'snr' }`
            Which variable to plot on the x axis

        ystr : `{ 'm_1', 'm_2', 'm_c', 'ecc', 'dist', 'f_orb', 'f_GW', 'a', snr' }`
            Which variable to plot on the y axis (if None then a 1D distribution is made using `xstr`)

        which_sources : `boolean array`
            Mask for which sources should be plotted (default is all sources)

        exclude_merged_sources : `boolean`
            Whether to exclude merged sources in distributions (default is
            True)

        **kwargs : `various`
            When only ``xstr`` is provided, the kwargs are the same as
            :meth:`legwork.visualisation.plot_1D_dist`. When both ``xstr`` and ``ystr`` are provided,
            the kwargs are the same as :meth:`legwork.visualisation.plot_2D_dist`.
            Note that if ``xlabel`` or ``ylabel`` is not passed then this function automatically creates
            one using a default string and (if applicable) the Astropy units of the variable.

        Returns
        -------
        fig : `matplotlib Figure`
            The figure on which the distribution is plotted

        ax : `matplotlib Axis`
            The axis on which the distribution is plotted
        """
        convert = {"m_1": self.m_1, "m_2": self.m_2, "m_c": self.m_c,
                   "ecc": self.ecc * u.dimensionless_unscaled, "dist": self.dist, "f_orb": self.f_orb,
                   "f_GW": self.f_orb * 2, "a": self.a,
                   "snr": self.snr * u.dimensionless_unscaled if self.snr is not None else self.snr}
        labels = {"m_1": "Primary Mass", "m_2": "Secondary Mass", "m_c": "Chirp Mass", "ecc": "Eccentricity",
                  "dist": "Distance", "f_orb": "Orbital Frequency", "f_GW": "Gravitational Wave Frequency",
                  "a": "Semi-major axis", "snr": "Signal-to-noise Ratio"}
        unitless = set(["ecc", "snr"])

        if which_sources is None:
            which_sources = np.repeat(True, self.n_sources)

        if exclude_merged_sources:
            which_sources = np.logical_and(which_sources, np.logical_not(self.merged))

        # ensure that the variable is a valid choice
        for var_str in [xstr, ystr]:
            if var_str not in convert.keys() and var_str is not None:
                error_str = "`xstr` and `ystr` must be one of: " \
                            + ', '.join(["`{}`".format(k) for k in list(convert.keys())])
                raise ValueError(error_str)

        # check the instance variable has been already set
        x = convert[xstr]
        if x is None:
            raise ValueError("x variable (`{}`)".format(xstr), "must be not be None")
        if ystr is not None:
            y = convert[ystr]
            if y is None:
                raise ValueError("y variable (`{}`)".format(ystr), "must be not be None")

        # create the x label if it wasn't provided
        if "xlabel" not in kwargs.keys():
            if xstr in unitless:
                kwargs["xlabel"] = labels[xstr]
            else:
                kwargs["xlabel"] = r"{} [{:latex}]".format(labels[xstr], x.unit)

        # create the y label if it wasn't provided and ystr was
        if ystr is not None and "ylabel" not in kwargs.keys():
            if ystr in unitless:
                kwargs["ylabel"] = labels[ystr]
            else:
                kwargs["ylabel"] = r"{} [{:latex}]".format(labels[ystr], y.unit)

        # work out what the weights are
        weights = self.weights[which_sources] if self.weights is not None else None

        # plot it!
        if ystr is not None:
            return vis.plot_2D_dist(x=x[which_sources].value, y=y[which_sources].value,
                                    weights=weights, **kwargs)
        else:
            return vis.plot_1D_dist(x=x[which_sources].value, weights=weights, **kwargs)

    def plot_sources_on_sc(self, snr_cutoff=0, fig=None, ax=None, show=True, **kwargs):  # pragma: no cover
        """Plot all sources in the class on the sensitivity curve

        Parameters
        ----------
        snr_cutoff : `float`
            SNR below which sources will not be plotted (default is to plot all sources)

        fig: `matplotlib Figure`
            A figure on which to plot the distribution. Both `ax` and `fig` must be supplied for either
            to be used

        ax: `matplotlib Axis`
            An axis on which to plot the distribution. Both `ax` and `fig` must be supplied for either
            to be used

        show : `boolean`
            Whether to immediately show the plot

        **kwargs : `various`
            Keyword arguments to be passed to plotting functions

        Returns
        -------
        fig : `matplotlib Figure`
            The figure on which the sources are plotted

        ax : `matplotlib Axis`
            The axis on which the sources are plotted

        Notes
        -----

        .. warning::

            Note that this function is not yet implemented for evolving sources.
            Evolving sources will not be plotted and a warning will be shown instead.
            We are working on implementing this soon!
        """
        # only allow plotting when an SNR has been calculated
        if self.snr is None:
            print("ERROR: No SNR has been calculated yet")
            return None, None

        detectable = self.snr > snr_cutoff
        inspiraling = np.logical_not(self.merged)

        # plot circular and stationary sources
        circ_stat = self.get_source_mask(circular=True, stationary=True)
        circ_stat = np.logical_and.reduce((circ_stat, detectable, inspiraling))
        if circ_stat.any():
            f_orb = self.f_orb[circ_stat]
            h_0_2 = self.get_h_0_n(2, which_sources=circ_stat).flatten()
            weights = self.weights[circ_stat] if self.weights is not None else None
            fig, ax = vis.plot_sources_on_sc_circ_stat(f_orb=f_orb, h_0_2=h_0_2, snr=self.snr[circ_stat],
                                                       weights=weights, snr_cutoff=snr_cutoff,
                                                       fig=fig, ax=ax, show=False,
                                                       label="Circular/Stationary",
                                                       **self._sc_params, **kwargs)

        # plot eccentric and stationary sources
        ecc_stat = self.get_source_mask(circular=False, stationary=True)
        ecc_stat = np.logical_and.reduce((ecc_stat, detectable, inspiraling))
        if ecc_stat.any():
            f_dom = self.f_orb[ecc_stat] * self.max_snr_harmonic[ecc_stat]
            weights = self.weights[ecc_stat] if self.weights is not None else None
            fig, ax = vis.plot_sources_on_sc_ecc_stat(f_dom=f_dom, snr=self.snr[ecc_stat], weights=weights,
                                                      snr_cutoff=snr_cutoff, show=show, fig=fig, ax=ax,
                                                      label="Eccentric/Stationary",
                                                      **self._sc_params, **kwargs)

        # show warnings for evolving sources
        circ_evol = self.get_source_mask(circular=True, stationary=False)
        circ_evol = np.logical_and.reduce((circ_evol, detectable, inspiraling))
        if circ_evol.any():
            print("{} circular and evolving".format(len(circ_evol[circ_evol])),
                  "sources detected, plotting not yet implemented for",
                  "evolving sources.")

        ecc_evol = self.get_source_mask(circular=True, stationary=False)
        ecc_evol = np.logical_and.reduce((ecc_evol, detectable, inspiraling))
        if ecc_evol.any():
            print("{} eccentric and evolving".format(len(ecc_evol[ecc_evol])),
                  "sources detected, plotting not yet implemented for",
                  "evolving sources.")

        return fig, ax


class Stationary(Source):
    """Subclass for sources that are stationary"""

    def get_snr(self, t_obs=4 * u.yr, instrument="LISA", custom_psd=None, verbose=False):
        self.snr = self.get_snr_stationary(t_obs=t_obs, instrument=instrument, custom_psd=custom_psd,
                                           verbose=verbose)
        return self.snr


class Evolving(Source):
    """Subclass for sources that are evolving"""

    def get_snr(self, t_obs=4 * u.yr, instrument="LISA", custom_psd=None, n_step=100, verbose=False):
        self.snr = self.get_snr_evolving(t_obs=t_obs, n_step=n_step, instrument=instrument,
                                         custom_psd=custom_psd, verbose=verbose)
        return self.snr


class VerificationBinaries(Source):
    """Generate a Source class with the LISA verification binaries preloaded.
    Data for the binaries is gathered from Kupfer+18 Table 1 and 2."""

    def __init__(self):
        # open file containing verification binary data
        with resources.path(package="legwork", resource="verification_binaries.npy") as path:
            vbs = np.load(path, allow_pickle=True)
            vbs = vbs.item()

        position = SkyCoord(l=vbs["l_gal"], b=vbs["b_gal"], distance=vbs["dist"], frame="galactic")

        # call the usual Source init function with this data
        super().__init__(m_1=vbs["m_1"], m_2=vbs["m_2"], dist=vbs["dist"],
                         f_orb=vbs["f_GW"].to(u.Hz) / 2, ecc=np.zeros(len(vbs["m_1"])),
                         position=position, inclination=vbs["i"])

        # also assign the labels and SNR
        self.labels = vbs["label"]
        self.true_snr = np.array(vbs["snr"])
