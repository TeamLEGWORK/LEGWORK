Scope and Limitations
=====================
LEGWORK is designed to provide quick estimates of the signal-to-noise ratio
of stellar-mass sources of mHz gravitational waves. These calculations can be
especially helpful when determining which sources in a large population containing
millions of binaries will be potentially detectable by LISA. If you are looking for
estimates of non-stellar-origin sources, check out the
`Gravitational Wave Universe Toolbox <http://gw-universe.org/>`_ or
`gwplotter <http://gwplotter.com/>`_. If you are looking for more advanced LISA simulator
tools, check out `lisacattools <https://github.com/tlittenberg/lisacattools>`_ or
`ldasoft <https://github.com/tlittenberg/ldasoft>`_.

The calculations done by LEGWORK apply the lowest-order post-Newtonian
description of gravitational wave emission and carefully follow the derivation of
`Flanagan and Hughes 1998a <https://ui.adsabs.harvard.edu/abs/1998PhRvD..57.4535F/abstract>`_.
This means that any higher order effects like spin-orbit
coupling or radiation reaction are not accounted for and thus if
a source is expected to depend on these higher order
effects the SNRs provided by LEGWORK may not be representative of the true SNR.
LEGWORK's SNRs are appropriate at mHz frequencies for sources with masses less than
a few tens of solar masses.

LEGWORK provides SNRs for two assumption cases: one where the detector orbit,
sky position of the source, and inclination of the source are all averaged and one where
the detector orbit is averaged while the sky position and inclination of of the source
are provided by the user following
`Cornish and Larson 2003 <https://ui.adsabs.harvard.edu/abs/2003PhRvD..67j3001C/abstract>`_.
The calculation which takes the sky position and inclination of the source into account
also accounts for frequency spreading due to doppler modulation from the detector's orbit.
This means that this calculation is only valid for circular sources and will *always* return
SNRs that are lower than the fully averaged SNR calculation because of the effects of the
frequency spreading.
 
