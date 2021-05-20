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