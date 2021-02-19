import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import astropy.units as u
import legwork.lisa as lisa

# set the default font and fontsize
plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.7 * fs,
          'ytick.labelsize': 0.7 * fs}
plt.rcParams.update(params)

__all__ = ['plot_1D_dist', 'plot_2D_dist', 'plot_sensitivity_curve',
           'plot_sources_on_sc_circ_stat', 'plot_sources_on_sc_ecc_stat']


def plot_1D_dist(x, weights=None, disttype="hist", fig=None, ax=None,
                 xlabel=None, ylabel=None, xlim=None, ylim=None, color=None,
                 show=True, **kwargs):
    """plot a 1D distribution of `x` with `weights`. This function is a
    wrapper for `matplotlib.pyplot.hist`, `seaborn.kdeplot` and
    `seaborn.ecdfplot`

    Parameters
    ----------
    x : `float/int array`
        variable to plot, should be a 1D array

    weights : `float/int array`
        weights for each variable in `x`, must have the same shape

    disttype : `{{ "hist", "kde", "ecdf" }}`
        which type of distribution plot to use

    fig: `matplotlib Figure`
        a figure on which to plot the distribution. Both `ax` and `fig` must be
        supplied for either to be used

    ax: `matplotlib Axis`
        an axis on which to plot the distribution. Both `ax` and `fig` must be
        supplied for either to be used

    xlabel : `string`
        label for the x axis, passed to Axes.set_xlabel()

    ylabel : `string`
        label for the y axis, passed to Axes.set_ylabel()

    xlim : `tuple`
        lower and upper limits for the x axis, passed to Axes.set_xlim()

    ylim : `tuple`
        lower and upper limits for the y axis, passed to Axes.set_ylim()

    color : `string or tuple`
        colour to use for the plot, see
        https://matplotlib.org/tutorials/colors/colors.html for details on how
        to specifiy a colour

    show : `boolean`
        whether to immediately show the plot or only return the Figure and Axis

    **kwargs : `(if disttype=="hist")`
        Include values for any of `bins, range, density, cumulative, bottom,
        histtype, align, orientation, rwidth, log, label`. See
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
        for more details.

    **kwargs : `(if disttype=="kde")`
        Include values for any of `gridsize, cut, clip, legend, cumulative,
        bw_method, bw_adjust, log_scale, fill, label, linewidth, linestyle`.See
        https://seaborn.pydata.org/generated/seaborn.kdeplot.html for more
        details.

    **kwargs : `(if disttype=="ecdf")`
        Include values for any of `stat, complementary, log_scale, legend,
        label, linewidth, linestyle`. See
        https://seaborn.pydata.org/generated/seaborn.ecdfplot.html
        for more details.

    Returns
    -------
    fig : `matplotlib Figure`
        the figure on which the distribution is plotted

    ax : `matplotlib Axis`
        the axis on which the distribution is plotted
    """
    # create new figure and axes is either weren't provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # possible kwargs for matplotlib.hist
    hist_args = {"bins": "auto", "range": None, "density": True,
                 "cumulative": False, "bottom": None, "histtype": 'bar',
                 "align": 'mid', "orientation": 'vertical', "rwidth": None,
                 "log": False, "label": None}

    # possible kwargs for seaborn.kdeplot
    kde_args = {"gridsize": 200, "cut": 3, "clip": None, "legend": True,
                "cumulative": False, "bw_method": 'scott', "bw_adjust": 1,
                "log_scale": None, "fill": None, "label": None,
                "linewidth": None, "linestyle": None}

    # possible kwargs for seaborn.ecdfplot
    ecdf_args = {"stat": 'proportion', "complementary": False,
                 "log_scale": None, "legend": True, "label": None,
                 "linewidth": None, "linestyle": None}

    # set which ones we are using for this plot
    plot_args = hist_args if disttype == "hist" else \
        kde_args if disttype == "kde" else ecdf_args

    # update the values with those supplied
    for key, value in kwargs.items():
        if key in plot_args:
            plot_args[key] = value
        else:
            # warn user if they give an invalid kwarg
            print("Warning: keyword argument `{}`".format(key),
                  "not recognised for disttype `{}`".format(disttype),
                  "and will be ignored")

    # create whichever plot was requested
    if disttype == "hist":
        ax.hist(x, weights=weights, color=color, **plot_args)
    elif disttype == "kde":
        sns.kdeplot(x=x, weights=weights, ax=ax, color=color,
                    **plot_args)
    elif disttype == "ecdf":
        sns.ecdfplot(x=x, weights=weights, ax=ax, color=color,
                     **plot_args)

    # update axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # update axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # immediately show the plot if requested
    if show:
        plt.show()

    # return the figure and axis for further plotting
    return fig, ax


def plot_2D_dist(x, y, weights=None, disttype="scatter", fig=None, ax=None,
                 xlabel=None, ylabel=None, xlim=None, ylim=None, color=None,
                 show=True, **kwargs):
    """plot a 2D distribution of `x` and `y` with `weights`. This function is a
    wrapper for `matplotlib.pyplot.scatter` and `seaborn.kdeplot`

    Parameters
    ----------
    x : `float/int array`
        variable to plot on the x axis, should be a 1D array

    y : `float/int array`
        variable to plot on the x axis, should be a 1D array

    weights : `float/int array`
        weights for each variable in `variable`, must have the same shape

    disttype : `{{ "scatter", "kde" }}`
        which type of distribution plot to use

    fig: `matplotlib Figure`
        a figure on which to plot the distribution. Both `ax` and `fig` must be
        supplied for either to be used

    ax: `matplotlib Axis`
        an axis on which to plot the distribution. Both `ax` and `fig` must be
        supplied for either to be used

    xlabel : `string`
        label for the x axis, passed to Axes.set_xlabel()

    ylabel : `string`
        label for the y axis, passed to Axes.set_ylabel()

    xlim : `tuple`
        lower and upper limits for the x axis, passed to Axes.set_xlim()

    ylim : `tuple`
        lower and upper limits for the u axis, passed to Axes.set_ylim()

    color : `string or tuple`
        colour to use for the plot, see
        https://matplotlib.org/tutorials/colors/colors.html for details on how
        to specifiy a colour

    show : `boolean`
        whether to immediately show the plot or only return the Figure and Axis

    **kwargs : `(if disttype=="scatter")`
        input any of `s, c, marker, cmap, norm, vmin, vmax, alpha, linewidths,
        edgecolors`. See
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
        for more details.

    **kwargs : `(if disttype=="kde")`
        input any of `gridsize, cut, clip, legend, cumulative, cbar, cbar_ax,
        cbar_kws, bw_method, hue, palette, hue_order, hue_norm, levels, thresh,
        bw_adjust, log_scale, fill, label`. See
        https://seaborn.pydata.org/generated/seaborn.kdeplot.html
        for more details.

    Returns
    -------
    fig : `matplotlib Figure`
        the figure on which the distribution is plotted

    ax : `matplotlib Axis`
        the axis on which the distribution is plotted
    """
    # create new figure and axes is either weren't provided
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # possible kwargs for matplotlib.hist
    scatter_args = {"s": None, "c": None, "marker": None, "cmap": None,
                    "norm": None, "vmin": None, "vmax": None, "alpha": None,
                    "linewidths": None, "edgecolors": None, "label": None}

    # possible kwargs for seaborn.kdeplot
    kde_args = {"gridsize": 200, "cut": 3, "clip": None, "legend": True,
                "cumulative": False, "cbar": False, "cbar_ax": None,
                "cbar_kws": None, "hue": None, "palette": None,
                "hue_order": None, "hue_norm": None, "levels": 10,
                "thresh": 0.05, "bw_method": 'scott', "bw_adjust": 1,
                "log_scale": None, "fill": None, "label": None}

    # set which ones we are using for this plot
    plot_args = scatter_args if disttype == "scatter" else kde_args

    # update the values with those supplied
    for key, value in kwargs.items():
        if key in plot_args:
            plot_args[key] = value
        else:
            # warn user if they give an invalid kwarg
            print("Warning: keyword argument `{}`".format(key),
                  "not recognised for disttype `{}`".format(disttype),
                  "and will be ignored")

    # create whichever plot was requested
    if disttype == "scatter":
        ax.scatter(x, y, color=color, **plot_args)
    elif disttype == "kde":
        sns.kdeplot(x=x, y=y, weights=weights, ax=ax, color=color, **plot_args)

    # update axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # update axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # immediately show the plot if requested
    if show:
        plt.show()

    # return the figure and axis for further plotting
    return fig, ax


def plot_sensitivity_curve(frequency_range=None, y_quantity="ASD", fig=None,
                           ax=None, show=True, color="#18068b", fill=True,
                           alpha=0.2, label=None, **kwargs):
    """plot the LISA sensitivity curve

    Parameters
    ----------
    frequency_range : `float array`
        frequency values at which to plot the sensitivity curve

    y_quantity : `{{ "ASD", "h_c" }}`
        which quantity to plot on the y axis (amplitude spectral density
        or characteristic strain)

    fig: `matplotlib Figure`
        a figure on which to plot the distribution. Both `ax` and `fig` must be
        supplied for either to be used

    ax: `matplotlib Axis`
        an axis on which to plot the distribution. Both `ax` and `fig` must be
        supplied for either to be used

    show : `boolean`
        whether to immediately show the plot or only return the Figure and Axis

    color : `string or tuple`
        colour to use for the curve, see
        https://matplotlib.org/tutorials/colors/colors.html for details on how
        to specifiy a colour

    fill : `boolean`
        whether to fill the area below the sensitivity curve

    alpha : `float`
        opacity of the filled area below the sensitivity curve (ignored if fill
        is `False`)

    label : `string`
        label for the sensitivity curve in legends

    **kwargs : `various`
        Keyword args are passed to :meth:`legwork.lisa.power_spectral_density`,
        see those docs for details on possible arguments.

    Returns
    -------
    fig : `matplotlib Figure`
        the figure on which the distribution is plotted

    ax : `matplotlib Axis`
        the axis on which the distribution is plotted
    """
    if frequency_range is None:
        frequency_range = np.logspace(-5, 0, 1000) * u.Hz

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # work out what the noise amplitude should be
    psd = lisa.power_spectral_density(f=frequency_range, **kwargs)
    if y_quantity == "ASD":
        noise_amplitude = np.sqrt(psd)
    elif y_quantity == "h_c":
        noise_amplitude = np.sqrt(frequency_range * psd)
    else:
        raise ValueError("y_quantity must be one of 'ASD' or 'h_c'")

    # plot the curve and fill if needed
    ax.loglog(frequency_range, noise_amplitude, color=color, label=label)
    if fill:
        ax.fill_between(frequency_range, 0, noise_amplitude, alpha=alpha,
                        color=color)

    # adjust labels, sizes and frequency limits to plot is flush to the edges
    ax.set_xlabel(r'Frequency [$\rm Hz$]')
    if y_quantity == "ASD":
        ax.set_ylabel(r'ASD $[\rm Hz^{-1/2}]$')
    else:
        ax.set_ylabel(r'Characteristic Strain')

    ax.tick_params(axis='both', which='major')
    ax.set_xlim(np.min(frequency_range).value, np.max(frequency_range).value)

    if show:
        plt.show()

    return fig, ax


def plot_sources_on_sc_circ_stat(f_orb, h_0_2, snr,
                                 snr_cutoff=0, t_obs=4 * u.yr,
                                 fig=None, ax=None, show=True, **kwargs):
    """overlay circular/stationary sources on the LISA sensitivty curve.
    Each source is plotted at its gravitational wave frequency (n = 2) such
    that its height above the curve is equal to it signal-to-noise ratio.

    Parameters
    ----------
    f_orb : `float/array`
        orbital frequency

    h_0_2 : `float/array`
        strain amplitude of the n = 2 harmonic

    snr : `float/array`
        signal-to-noise ratio

    snr_cutoff : `float`
        SNR above which to plot binaries (default is 0 such that all
        sources are plotted)

    t_obs : `float`
        LISA observation time

    fig: `matplotlib Figure`
        a figure on which to plot the distribution. Both `ax` and `fig`
        must be supplied for either to be used

    ax: `matplotlib Axis`
        an axis on which to plot the distribution. Both `ax` and `fig`
        must be supplied for either to be used

    show : `boolean`
        whether to immediately show the plot or only return the Figure
        and Axis

    **kwargs : `various`
        This function is a wrapper on `visualisation.plot_2D_dist` and each
        kwarg is passed directly to this function. For example, you can write
        `disttype="kde"` for a kde density plot instead of a scatter plot.

    Returns
    -------
    fig : `matplotlib Figure`
        the figure on which the distribution is plotted

    ax : `matplotlib Axis`
        the axis on which the distribution is plotted
    """
    # create figure if it wasn't provided
    if fig is None or ax is None:
        fig, ax = plot_sensitivity_curve(show=False, t_obs=t_obs)

    # work out which binaries are above the cutoff
    detectable = snr > snr_cutoff
    if not detectable.any():
        print("ERROR: There are no binaries above provided `snr_cutoff`")
        return fig, ax

    # calculate the GW frequency and ASD for detectable binaries
    f_GW = f_orb[detectable] * 2
    asd = ((1/4 * t_obs)**(1/2) * h_0_2[detectable]).to(u.Hz**(-1/2))

    # plot either a scatter or density plot of the detectable binaries
    ylims = ax.get_ylim()
    fig, ax = plot_2D_dist(x=f_GW, y=asd, fig=fig, ax=ax, show=False, **kwargs)
    ax.set_ylim(ylims)

    if show:
        plt.show()

    return fig, ax


def plot_sources_on_sc_ecc_stat(f_dom, snr, snr_cutoff=0, t_obs=4 * u.yr,
                                fig=None, ax=None, show=True, **kwargs):
    """overlay eccentric/stationary sources on the LISA sensitivty curve.
    Each source is plotted at its dominant harmonic frequency such that
    that its height above the curve is equal to it signal-to-noise ratio.

    Parameters
    ----------
    f_dom : `float/array`
        dominant harmonic frequency (f_orb * n_dom where n_dom is the harmonic
        with the maximum strain)

    snr : `float/array`
        signal-to-noise ratio

    snr_cutoff : `float`
        SNR above which to plot binaries (default is 0 such that all
        sources are plotted)

    fig: `matplotlib Figure`
        a figure on which to plot the distribution. Both `ax` and `fig`
        must be supplied for either to be used

    ax: `matplotlib Axis`
        an axis on which to plot the distribution. Both `ax` and `fig`
        must be supplied for either to be used

    show : `boolean`
        whether to immediately show the plot or only return the Figure
        and Axis

    **kwargs : `various`
        This function is a wrapper on `visualisation.plot_2D_dist` and each
        kwarg is passed directly to this function. For example, you can write
        `disttype="kde"` for a kde density plot instead of a scatter plot.

    Returns
    -------
    fig : `matplotlib Figure`
        the figure on which the distribution is plotted

    ax : `matplotlib Axis`
        the axis on which the distribution is plotted
    """
    # create figure if it wasn't provided
    if fig is None or ax is None:
        fig, ax = plot_sensitivity_curve(show=False, t_obs=t_obs)

    # work out which binaries are above the cutoff
    detectable = snr > snr_cutoff
    if not detectable.any():
        print("ERROR: There are no binaries above provided `snr_cutoff`")
        return fig, ax

    # calculate asd that makes it so height above curve is snr
    asd = snr[detectable] \
        * np.sqrt(lisa.power_spectral_density(f_dom[detectable]))

    # plot either a scatter or density plot of the detectable binaries
    ylims = ax.get_ylim()
    fig, ax = plot_2D_dist(x=f_dom[detectable], y=asd.to(u.Hz**(-1/2)),
                           fig=fig, ax=ax, show=False, **kwargs)
    ax.set_ylim(ylims)

    if show:
        plt.show()

    return fig, ax
