---
title: 'LEGWORK: A python package for computing the evolution and detectability of stellar-origin gravitational-wave sources with space-based detectors'
tags:
  - Python
  - astronomy
  - gravitational waves
  - gravitational wave detectors
  - LISA
  - TianQin
  - compact objects
  - orbital evolution
  - white dwarfs
  - neutron stars
  - stellar mass black holes
authors:
  - name: Tom Wagg
    orcid: 0000-0001-6147-5761
    affiliation: "1, 2, 3"
  - name: Katelyn Breivik
    orcid: 0000-0001-5228-6598
    affiliation: 4
  - name: Selma E. de Mink
    orcid: 0000-0001-9336-2825
    affiliation: "3, 5, 2"
affiliations:
 - name: Department of Astronomy, University of Washington, Seattle, WA, 98195
   index: 1
 - name: Center for Astrophysics | Harvard & Smithsonian, 60 Garden Street, Cambridge, MA 02138, USA
   index: 2
 - name: Max-Planck-Institut für Astrophysik, Karl-Schwarzschild-Straße 1, 85741 Garching, Germany
   index: 3
 - name: Center for Computational Astrophysics, Flatiron Institute, 162 Fifth Ave, New York, NY, 10010, USA
   index: 4
 - name: Anton Pannekoek Institute for Astronomy and GRAPPA, University of Amsterdam, NL-1090 GE Amsterdam, The Netherlands
   index: 5
date: 10 November 2021
bibliography: paper.bib
---

\newcommand{\unit}[1]{\, \mathrm{#1}}
\newcommand{\lw}{LEGWORK}

# Summary

\lw{} (LISA Evolution and Gravitational Wave Orbit Kit) is an open-source Python package for making predictions about stellar-origin gravitational wave sources and their detectability in LISA or other space-based gravitational wave detectors. \lw{} can be used to evolve the orbits of sources due to gravitational wave emission, calculate gravitational wave strains (using post-Newtonian approximations), compute signal-to-noise ratios and visualise the results. It can be applied to a variety of potential sources, including binaries consisting of white dwarfs, neutron stars and black holes. Although we focus on double compact objects, in principle \lw{} can be used for any system with a user-specified orbital evolution, such as those affected by a third object or gas drag. We optimised the package to make it efficient for use in population studies which can contain tens-of-millions of sources. We hope that \lw{} will enable and accelerate future studies triggered by the rapidly growing interest in gravitational wave sources.

# Statement of need

The planned space-based gravitational wave detector LISA (Laser Interferometer Space Antenna, @Amaro-Seoane+2017) will present an entirely new view of gravitational waves by focusing on lower frequencies ($10^{-5} < f / \unit{Hz} < 10^{-1}$) than ground-based detectors. This will enable the study of many new source classes including mergers of supermassive black holes e.g.[@Begelman+1980; @Klein+2016; @Bellovary2019], extreme mass ratio inspirals e.g.[@Berti2006; @Barack2007; @Babak2017; @Moore2017], and cosmological GW backgrounds e.g.[@Bartolo2016; @Caprini2016; @Caldwell2019]. However, this frequency regime is also of interest for the detection of local stellar-mass binaries during their inspiral phase. LISA is expected to detect Galactic stellar-origin binaries containing combinations of white dwarfs, neutron stars, and black holes, ranging from the numerous double white dwarf population, to the rare but loud double black hole population.

The potential to detect stellar-origin sources with LISA has been studied in the past e.g.[@Nelemans+2001; @Liu+2009; @Liu+2014; @Ruiter+2010; @Belczynski+2010; @Nissanke+2012]. More recently, the direct detection of gravitational waves with ground-based detectors has led to renewed interest in this topic with many recent papers addressing the issue e.g.[@Christian+2017; @Kremer+2017; @Kremer+2018; @Korol+2017; @Korol+2018; @Korol+2019; @Korol+2020; @Lamberts+2018; @Lamberts+2019; @Fang+2019; @Andrews+2020; @Lau+2020; @Breivik+2020; @Breivik+2020a; @Roebber+2020; @Chen+2020; @Sesana+2020; @Shao+2021].

Each of these studies require making estimates of the signal-to-noise ratio of individual binary systems and possibly the slow gravitational wave inspiral that lead to the present-day parameters. So far, most studies made use of custom made codes which have not been made publicly available. 

We believe that the large renewed interest in LISA and the stellar-origin sources it may detect will lead to many more studies in the near future that would need similar computations. This leads to a significant amount of redundancy which, at best results in extra work for each individual and at worst leads to an increased chance of introducing mistakes and inconsistencies when translating the necessary expressions to software.

\lw{} is a Python package designed to streamline the process of making predictions of LISA detection rates for stellar-origin binaries such that it is as fast, reliable and simple as possible. This goal makes \lw{} unique among other implementations of gravitational-wave tools in the literature, which focus on a more broad coverage of the gravitational-wave spectrum and source classes, rather than an optimised approach for certain sources e.g.[@Moore+2015; @Yi2021]. With \lw{} one can evolve the orbits of a binary or a collection of binaries and calculate their strain amplitudes for any range of frequency harmonics. One can compute the sensitivity curve for LISA or other future gravitational wave detectors (e.g. TianQin's curve, or that of a custom instrument) and use it to compute the signal-to-noise ratio of a collection of sources. Furthermore, \lw{} provides tools to visualise all of the results with easy-to-use plotting functions. Finally, \lw{} is fully tested to check for consistency in the derivations described below.

Specifically, we implement the expressions by @Peters+1963 and @Peters+1964 for the evolution of binary orbits due to the emission of gravitational waves, equations for the strain amplitudes and signal-to-noise ratios of binaries from various papers [e.g. @Flanagan+1998; @Finn+2000; @Cornish2003; @Barack+2004; @Moore+2015] and approximations for the LISA and TianQin sensitivity curves given in @Robson+2019 and @Huang+2020 respectively.

The open-source nature of the project means that new users as well as seasoned experts in the field can work together in a collaborative setting to consider new features and enhancements to the package as well as check the implementation. At the same time, with our thorough online documentation, derivations and tutorials, we hope \lw{} can make this functionality more accessible to the broader scientific community.

# Acknowledgements
We are grateful to Stas Babak, Floor Broekgaarden, Tom Callister, Will Farr, Stephen Justham, Valeria Korol, Mike Lau, Tyson Littenberg, Ilya Mandel, Alberto Sesana, Lieke van Son, the CCA GW group, the BinCosmos group and the COMPAS group for stimulating discussions that influenced and motivated us to complete this project. We thank the BinCosmos group for testing an early version of the package and providing useful feedback. In particular, we thank Lieke van Son for her innovation in inventing the name \lw{}! TW also thanks Floor Broekgaarden for first suggesting that he investigate LISA and the derivation of the SNR calculation.
    
This project was funded in part by the National Science Foundation under Grant No.\ (NSF grant number 2009131), the European Union’s Horizon 2020 research and innovation program from the European Research Council (ERC, Grant agreement No.\ 715063), and by the Netherlands Organization for Scientific Research (NWO) as part of the Vidi research program BinWaves with project number 639.042.728. We further acknowledge the Black Hole Initiative funded by a generous contribution of the John Templeton Foundation and the Gordon and Betty Moore Foundation. The Flatiron Institute is funded by the Simons Foundation.

# References