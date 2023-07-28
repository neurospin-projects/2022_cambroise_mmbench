# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Plotting utility functions.
"""

# Imports
import os
import math
from itertools import combinations
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind as ttest
from mmbench.color_utils import print_subtitle, print_result
from itertools import cycle


def plot_barrier_clusters(data, labels, scores, task_name, metric_name):
    """ Display the barrier clustering results.

    Parameters
    ----------
    data: (N, t)
        the time courses obtained when interpolating the weights of any two
        pairs of intances.
    labels: (N, )
        the time courses clusters.
    scores: (K, )
        the clustering metrics used to determine to best number of clusters.
    task_name: str
        the task name used in the barrier expermiement.
    metric_name: str
        metric name used to select the best number of clusters.
    """
    fontparams = {"font.size": 11, "font.weight": "bold",
                  "font.family": "serif", "font.style": "italic"}
    plt.rcParams.update(fontparams)
    labelparams = {"size": 16, "weight": "semibold", "family": "serif"}
    unique_labels = np.unique(labels)
    n_cluster = len(unique_labels)
    max_clusters = len(scores)
    alpha = range(data.shape[-1])
    xk = range(1, max_clusters + 1)
    cmap = cm.get_cmap("hsv", max_clusters)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    for label in unique_labels:
        ts = data[labels == label]
        mean_ts = np.mean(ts, axis=0)
        std_ts = np.std(ts, axis=0)
        ax.plot(alpha, mean_ts, label=f"basin {label + 1}", c=cmap(label))
        ax.fill_between(alpha, mean_ts - std_ts, mean_ts + std_ts, alpha=0.3,
                        facecolor=cmap(label))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel(r"$\alpha$", labelparams)
    ax.set_ylabel(task_name, labelparams)
    handles, labels = ax.get_legend_handles_labels()
    kw = dict(ncol=len(handles), loc="lower center", frameon=False)
    leg = ax.legend(handles, labels, bbox_to_anchor=[0.5, 1.04], **kw)
    ax.add_artist(leg)
    fig.subplots_adjust(top=0.9)
    plt.subplot(1, 2, 2)
    ax = plt.gca()
    ax.plot(xk, scores)
    plt.vlines(n_cluster, plt.ylim()[0], plt.ylim()[1], linestyles="dashed")
    plt.text(n_cluster, (plt.ylim()[0] + plt.ylim()[1]) / 2, f"k={n_cluster}",
             ha="center", va="center", rotation="vertical",
             backgroundcolor="white")
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlabel("k", labelparams)
    ax.set_ylabel(metric_name, labelparams)
    return fig


def plot_mat(key, mat, ax=None, figsize=(5, 2), dpi=300, fontsize=16,
             fontweight="bold", title=None, vmin=None, vmax=None):
    """ Display a mat array.

    Parameters
    ----------
    key: str
        the mat array identifier.
    mat: array (n, n)
        the mat array to display.
    ax: matplotlib.axes.Axes, default None
        the axes used to display the plot.
    figsize: (float, float), default (5, 2)
        width, height in inches.
    dpi: float, default 300
        the resolution of the figure in dots-per-inch.
    fontsize: int or str, default 16
        size in points or relative size, e.g., 'smaller', 'x-large'.
    fontweight: str, default 'bold'
        the font weight, e.g. 'normal', 'bold', 'heavy', 'light', 'ultrabold'
        or 'ultralight'.
    title: str, default None
        the title displayed on the figure.
    vmin: float, default None
        minimum value on y-axis of figures.
    vmax: float, default None
        maximum value on y-axis of figures.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.imshow(mat, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
    if title is None:
        plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)


def plot_bar(key, rsa, ax=None, figsize=(5, 2), dpi=300, fontsize=16,
             fontsize_star=25, fontweight="bold", line_width=2.5,
             marker_size=.1, title=None, palette="Spectral", report_t=False,
             do_pairwise_stars=False, do_one_sample_stars=True,
             yname="model fit (r)"):
    """ Display results with bar plots.

    Parameters
    ----------
    key: str
        the analysis identifier.
    rsa: dict of dict
        the sampling data with the analysis identifiers as first key and
        experimental conditions as second key.
    ax: matplotlib.axes.Axes, default None
        the axes used to display the plot.
    figsize: (float, float), default (5, 2)
        width, height in inches.
    dpi: float, default 300
        the resolution of the figure in dots-per-inch.
    fontsize: int or str, default 16
        size in points or relative size, e.g., 'smaller', 'x-large'.
    fontsize_star: int or str, default 25
        size in points or relative size, e.g., 'smaller', 'x-large' used to
        display pairwise statistics.
    fontweight: str, default 'bold'
        the font weight, e.g. 'normal', 'bold', 'heavy', 'light', 'ultrabold'
        or 'ultralight'.
    line_width: int, default 2.5
        the bar plot line width.
    marker_size: int, default .1
        the sampling scatter plot marker size.
    title: str, default None
        the title displayed on the figure.
    palette: palette name, list, or dict
        colors to use for the different levels of the hue variable.
        Should be something that can be interpreted by color_palette(), or a
        dictionary mapping hue levels to matplotlib colors.
    report_t: bool, default False
        optionally, generate a report with pairwise statistics.
    do_pairwise_stars: bool, default False
        optionally, display pairwise statistics.
    do_one_sample_stars: bool, default True
        optionally, display sampling statistics.
    yname: str, default 'model fit (r)'
        optionally, name of the metric on y-axis.

    Returns
    -------
    pairwise_stat_df: pandas.DataFrame or None
        the generated pairwise statistics.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    data = rsa[key]
    _data = {}
    for cond in list(data.keys()):
        _data.setdefault("model fit (r)", []).extend(data[cond])
        _data.setdefault("condition", []).extend([cond] * len(data[cond]))
    data_df = pd.DataFrame.from_dict(_data)

    sns.stripplot(data=data_df,
                  x="condition",
                  y="model fit (r)",
                  jitter=0.15,
                  alpha=1.0,
                  color="k",
                  size=marker_size)
    plot = sns.barplot(data=data_df,
                       x="condition",
                       y="model fit (r)",
                       errcolor="r",
                       alpha=0.3,
                       linewidth=line_width,
                       errwidth=line_width,
                       palette=palette)
    for patch in plot.containers[0]:
        fc = patch.get_edgecolor()
        patch.set_edgecolor(mcolors.to_rgba(fc, 1.))
    locs, labels = plt.yticks()
    new_y = locs
    new_y = np.linspace(locs[0], locs[-1], 6)
    plt.yticks(new_y, labels=[f"{yy:.2f}" for yy in new_y], fontsize=fontsize,
               fontweight=fontweight)
    plt.ylabel(yname, fontsize=fontsize, fontweight=fontweight)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(line_width)
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    _xlabels = ["\n".join(item.split("_")[:-1]) for item in xlabels]
    ax.set_xticklabels(_xlabels, fontsize=fontsize, fontweight=fontweight)
    x_label = ax.axes.get_xaxis().get_label()
    x_label.set_visible(False)
    ylim = plt.ylim()
    plt.ylim(np.array(ylim) * (1, 1.1))
    if title is None:
        plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)

    if do_one_sample_stars:
        one_sample_thresh = np.array((1, .05, .001, .0001))
        one_sample_stars = np.array(("n.s.", "*", "**", "***"))
        for idx, name in enumerate(xlabels):
            one_sample = ttest_1samp(data[name], 0)
            these_stars = one_sample_stars[
                max(np.nonzero(one_sample.pvalue < one_sample_thresh)[0])]
            _xlabels[idx] = f"{_xlabels[idx]}\n({these_stars})"
        ax.set_xticklabels(_xlabels, fontsize=fontsize, fontweight=fontweight)

    if report_t or do_pairwise_stars:
        size = len(xlabels)
        pairwise_t = np.zeros((size, size))
        pairwise_p = np.zeros((size, size))
        _data = dict()
        for idx1, name1 in enumerate(xlabels):
            for idx2, name2 in enumerate(xlabels):
                n_samples = len(data[name1])
                tval, pval = ttest(data[name1], data[name2])
                if pval > .001:
                    print(f"{key} {name1} >  {name2} | "
                          f"t({n_samples-1}) = {tval:.2f} p = {pval:.2f}")
                else:
                    print(f"{key} {name1} >  {name2} | "
                          f"t({n_samples-1}) = {tval:.2f} p < .001")
                pairwise_t[idx1, idx2] = tval
                pairwise_p[idx1, idx2] = pval
                _data.setdefault("pair", []).append(
                    f"qname-{key}_src-{name1.replace('_', '-')}_"
                    f"dest-{name2.replace('_', '-')}")
                _data.setdefault("tval", []).append(tval)
                _data.setdefault("pval", []).append(pval)
        pairwise_stat_df = pd.DataFrame.from_dict(_data)
    else:
        pairwise_stat_df = None

    if do_pairwise_stars:
        from statannotations.Annotator import Annotator
        pairwise_sample_thresh = np.array((1, .05, .001, .0001))
        pairwise_sample_stars = np.array(("n.s.", "*", "**", "***"))
        comps = list(combinations(range(len(xlabels)), 2))
        pairs, annotations = [], []
        for comp_idx, this_comp in enumerate(comps):
            sig_idx = max(np.nonzero(
                pairwise_p[this_comp[0], this_comp[1]] <
                pairwise_sample_thresh)[0])
            if sig_idx != 0:
                stars = pairwise_sample_stars[sig_idx]
                pairs.append([xlabels[this_comp[0]], xlabels[this_comp[1]]])
                annotations.append(stars)
        if len(pairs) > 0:
            annotator = Annotator(
                ax, pairs, data=data_df, x="condition", y="model fit (r)",
                order=xlabels)
            annotator.set_custom_annotations(annotations)
            annotator.annotate()

    return pairwise_stat_df


def barrier_display(coeffs, l_metrics, model_name, downstream, dataset, outdir,
                    scale, sname, color=None):
    """ Save barrier curves for a model.

    Parameters
    ----------
    coeffs : list
        the abscissa of the graph.
    l_metrics : array (n, n, n_coeffs)
        value matrix of the curve between two models.
    model_name : str
        name of the model.
    downstream : str
        name of the downstream task.
    dataset: str
        the dataset name: euaims or hbn.
    outdir : str
        the destination folder.
    scale : tuple (min, max)
        min and max values of matrix in matrices.
    sname : str
        the name of the scorer.
    """
    print_subtitle(f"Display {model_name}_{downstream} figures...")
    ncols = 3
    nrows = 4
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    for idx, row in enumerate(l_metrics):
        ax = plt.subplot(nrows, ncols, idx + 1)
        plot_curve(
            coeffs, row, ax=ax, figsize=None, dpi=300, fontsize=7,
            fontweight="bold", title=f"from run {idx + 1}", color=color)
        ax.set_ylim(scale[0], scale[1])
        ax.set_ylabel(sname)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=1, hspace=.5)
    plt.suptitle(f"{model_name} {downstream} {dataset} BARRIER FIGURES",
                 fontsize=20, y=.95)
    filename = os.path.join(outdir,
                            f"barrier_{model_name}_{downstream}_{dataset}.png")
    plt.savefig(filename)
    print_result(f"BARRIER: {filename}")


def mat_display(matrices, dataset, outdir, downstream_name, scale):
    """ Plot area matrices.

    Parameters
    ----------
    matrices : dict
        area matrix dictionaries by models.
    dataset: str
        the dataset name: euaims or hbn.
    outdir : str
        the destination folder.
    downstream_name: str
        the name of the column that contains the downstream classification
        task.
    scale : tuple (min, max)
        min and max values of matrix in matrices.
    """
    ncols = 2
    nrows = 3
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    for idx, key in enumerate(matrices):
        size = math.sqrt(matrices[key].size)
        ax = plt.subplot(nrows, ncols, idx + 1)
        plot_mat(
            key, matrices[key], ax=ax, figsize=None, dpi=300, fontsize=7,
            fontweight="bold", title=f"{key}", vmin=scale[0], vmax=scale[1])
        ax.set_xticks(np.arange(0, size, 2))
        ax.set_yticks(np.arange(0, size, 2))
        ax.set_xticklabels(np.arange(1, size + 1, 2))
        ax.set_yticklabels(np.arange(1, size + 1, 2))
        plt.colorbar(ax.images[0], ax=ax)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{downstream_name} {dataset} BARRIER AREA", fontsize=20,
                 y=.95)
    filename = os.path.join(outdir,
                            f"barrier_area_{downstream_name}_{dataset}.png")
    plt.savefig(filename)
    print_result(f"AREA: {filename}")


def plot_curve(xticks, mat, ax=None, figsize=(5, 2), dpi=300, fontsize=16,
               fontweight="bold", title=None, lcolor=None):
    """ Display a list of curve.

    Parameters
    ----------
    xticks: list
        the list of xtick locations.
    mat: array (n_curve, n_points)
        the matrix containing the points of the curves.
    ax: matplotlib.axes.Axes, default None
        the axis used to display the plot.
    figsize: (float, float), default (5, 2)
        width, height in inches.
    dpi: float, default 300
        the resolution of the figure in dots-per-inch.
    fontsize: int or str, default 16
        size in points or relative size, e.g., 'smaller', 'x-large'.
    fontweight: str, default 'bold'
        the font weight, e.g. 'normal', 'bold', 'heavy', 'light', 'ultrabold'
        or 'ultralight'.
    title: str, default None
        the title displayed on the figure.
    """
    c_default = cycle(plt.rcParams["axes.prop_cycle"])
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if lcolor is None:
        lcolor = list(range(mat.shape[0]))
    for idx, elem in enumerate(mat):
        color = [c["color"] for c in c_default][lcolor[idx]]
        ax.plot(xticks, elem, label=f"to {idx+1}", color=color)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if title is None:
        plt.title(xticks, fontsize=fontsize * 1.5, pad=2,
                  fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
