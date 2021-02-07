import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_1D_dist(variable, weights=None, disttype="hist", fig=None, ax=None,
                 xlabel="", ylabel="", xlim=None, ylim=None, color=None,
                 show=True, **kwargs):
    """plot a 1D distribution of `variable` with `weights`. This function is a
    wrapper for `matplotlib.pyplot.hist`, `seaborn.kdeplot` and
    `seaborn.ecdfplot`

    Params
    ------
    variable : `float/int array`
        variable to plot, should be a 1D array

    weights : `float/int array`
        weights for each variable in `variable`, must have the same shape

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
        lower and upper limits for the u axis, passed to Axes.set_ylim()

    color : `string or tuple`
        colour to use for the plot, see
        https://matplotlib.org/tutorials/colors/colors.html for details on how
        to specifiy a colour

    show : `boolean`
        whether to immediately show the plot or only return the Figure and Axis

    Keyword Args
    ------------
    The keyword args in this function are passed to the respective function
    that is used (set by `disttype`). Depending on your choice of `disttype`,
    there are many options:

    For `disttype="hist"` : bins, range, density, cumulative, bottom, histtype,
                            align, orientation, rwidth, log, label
    See https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.hist.html
    for more details.

    For `disttype="kde"` : gridsize, cut, clip, legend, cumulative, bw_method,
                           bw_adjust, log_scale, fill, label
    See https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    for more details.

    For `disttype="ecdf"` : stat, complementary, log_scale, legend, label
    See https://seaborn.pydata.org/generated/seaborn.ecdfplot.html
    for more details.

    Returns
    -------
    fig : `matplotlib Figure`
        the figure on which the distribution is plotted

    fig : `matplotlib Axis`
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
                "log_scale": None, "fill": None, "label": None}

    # possible kwargs for seaborn.ecdfplot
    ecdf_args = {"stat": 'proportion', "complementary": False,
                 "log_scale": None, "legend": True, "label": None}

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
        ax.hist(variable, weights=weights, color=color, **plot_args)
    elif disttype == "kde":
        sns.kdeplot(x=variable, weights=weights, ax=ax, color=color,
                    **plot_args)
    elif disttype == "ecdf":
        sns.ecdfplot(x=variable, weights=weights, ax=ax, color=color,
                     **plot_args)

    # update axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # update axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # immediately show the plot if requested
    if show:
        plt.show()

    # return the figure and axis for further plotting
    return fig, ax
