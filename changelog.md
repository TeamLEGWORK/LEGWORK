# LEGWORK Changelog
This log keeps track of the changes implemented in each version of LEGWORK.

## 0.0.1
- Initial release

## 0.0.2
- Update version number to work with pip

## 0.0.3
*TW, 12/04/21*
- Allow the computation of merger times with ``get_merger_times``
    - After computing, times are used automatically
    in subsequent SNR calculations to avoid doubling up the computation
- Add ``evolve_sources`` function that evolves sources through time, updates merger times and if necessary marks them as merged
    - Merged sources are ignored in other computations (e.g. strain and SNR)
- Add ``ret_snr2_by_harmonic`` to eccentric snr functions to allow the user to get the SNR at each harmonic separately instead of the total
- Add minor fixes to snr for evolving sources for when sources are closes to merging

## 0.0.4
*TW, 19/05/21*
- Change visualisation module to be more flexible with **kwargs (allow any for dist plot and add linewidth to sensitivity curve function)
- Change Source.get_snr() to allow re-interpolation of the sensitivity curve for convenience (and fix the warning so it works properly)

## 0.0.5
*TW, 25/09/21*
- Avoid LSODA warnings by preventing integration from getting near the singularity at the merger
- Allow user to select how long before a merger to stop integration

## 0.0.6
*TW, 26/10/21*
- Avoid plotting merged sources in any of the automatic routines
- Allow source class evolution code to handle sources close to their merger
- Ensure SNR calculation works if some sources have merged and produces no warnings
- Change default behaviour of Source class with interpolate_g - no longer always interpolate, only when the collection of sources is fairly large or it contains eccentric sources
- Add a warning for if all timesteps are too close to the merger (based on `t_before`) and hence evolution can't happen

## 0.1.0
*TW, KB 31/10/21*
Major version change as we've added a significant enhancement with the new non-average SNR calculations.

- Change `snr` module to allow the calculation of non-averaged SNR using exact inclination, sky position and polarisations
- Let users specific inclination, sky position and polarisation in `Source` instantiation
- Add `VerificationBinaries` class to `Source` module for convenient access to LISA verification binary data from Kupfer+18
- Change max line length in code from 80 to 110 to increase readability

## 0.1.1
*TW, 5/11/21*
Small changes to visualisation code and updates to tutorials/demos with the new code
- allow user to specify weights in `Source` and visualisation functions
- allow user to customise sensitivity curve in any use of `plot_sources_on_sc`
- `legwork.__version__` now prints the version number
- Add new demo about verification binaries and other miscellaneous docs fixes

## 0.1.2
*TW 12/11/21*
- Change default values for `small_e_tol` and `large_e_tol` in `get_t_merge_ecc`
- allow users to specify custom confusion noise in sensitivity curves

## 0.1.3
*TW 16/11/21*
- Move `determine_stationarity` from `utils` to `evol` to avoid cyclical imports

## 0.1.4
*TW 17/11/21*
- Add `custom_psd` to the `sc_params` in `Source`
- Change the default parameters of `Source.get_snr()` to use `sc_params`

## 0.1.5
*KB 18/11/21*
- Make confusion noise follow shape of supplied frequency even in case of no confusion

## 0.1.6
*TW 18/11/21*
- Add a factor of 10/3 to the TianQin sensitivity curve to make it consistent with LISA (thanks to Yi-Ming Hu for pointing this out!)

## 0.1.7
*TW 05/01/21*
- Link GitHub releases to Zenodo

## 0.2.0
*TW 05/01/21*
- A couple of changes to how confusion noise is handled
    - End user can now access confusion noise functions directly through `get_confusion_noise`
    - Added confusion noise fit from Huang+20 to be used with TianQin
    - Added confusion noise fit from Thiele+21 which is based on a WDWD population with a metallicity dependent binary fraction
- TianQin psd function now include the confusion noise
- Change defaults used in `source` and `psd`
    - Often defaults for arm length, observation time and confusion noise were previously LISA related, LEGWORK now automatically works out the defaults based on what instrument is chosen
- Bug fix: in `visualisation` avoid mixing floats with Quantities when filling in a sensitivity curve

## 0.2.1
*TW 11/01/22*
- [Issue [#64](https://github.com/TeamLEGWORK/LEGWORK/issues/64)] Remove "auto" option from `interpolate_g` in favour of interpolating by default warning the user if they don't have many samples
- [Issue [#76](https://github.com/TeamLEGWORK/LEGWORK/issues/76)] Make it so that `strain.h_0_n` returns as unitless (same as `source.Source.get_h_0_n`). Same for `snr` functions.
- Fixed an issue introduced in 0.2.0 where automated observation times didn't work in `visualisation.plot_sources_on_sc_circ_stat`
- [Issue [#78](https://github.com/TeamLEGWORK/LEGWORK/issues/78)] Add a warning for when people are evolving past the merger with `avoid_merger=True`

*KB 13/01/22*
- [Issues [#79](https://github.com/TeamLEGWORK/LEGWORK/issues/79), [#80](https://github.com/TeamLEGWORK/LEGWORK/issues/80)] Add discussion of limitiations and scope of legwork snr calculations

## 0.2.2
*KB 18/01/22*
- [Issues [#68](https://github.com/TeamLEGWORK/LEGWORK/issues/68), [#84](https://github.com/TeamLEGWORK/LEGWORK/issues/84)] Removes upper bound on version limits for numpy and numba

## 0.2.3
*TW 23/01/22*
- [Issue [#75](https://github.com/TeamLEGWORK/LEGWORK/issues/75)] Fix mixing quantities with floats when plotting
- [Issue [#86](https://github.com/TeamLEGWORK/LEGWORK/issues/86)] Clarify how notebooks should be run and update the installation instructions

## 0.2.4
*TW 27/01/22*
- Make dependencies in setup.cfg match requirements.txt!

## 0.2.5
*TW 27/01/22*
- [Issue [#89](https://github.com/TeamLEGWORK/LEGWORK/issues/89)]
    - Created environment.yml for the package
    - Updated installation instructions to match
- [Issue [#90](https://github.com/TeamLEGWORK/LEGWORK/issues/90)] Added all psd functions to __all__

## 0.3.0
*TW 01/02/22*
- New major version of LEGWORK after several updates during the JOSS review (see 0.2.0-0.2.5)
