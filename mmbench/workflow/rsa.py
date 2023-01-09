# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the Representational Similarity Analysis (RSA) workflows.
"""

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mmbench.stat_utils import data2mat, vec2mat, fit_rsa
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result,
    print_error)
import matplotlib.pyplot as plt
from mmbench.plotting import plot_mat, plot_bar


def benchmark_rsa_exp(dataset, datasetdir, outdir):
    """ Compare the learned latent space of different models using
    Representational Similarity Analysis (RSA).

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.

    Note
    ----
    The samples are generated with the 'bench-latent' sub-command and are
    stored in the 'outdir' in a file named 'latent_vecs.npz'. The samples
    shape is (n_samples, n_subjects, latent_dim). All samples must have the
    same number of samples and subjects, but possibly different latent
    dimensions.
    """
    print_title(f"COMPARE MODELS USING RSA ANALYSIS: {dataset}")
    benchdir = outdir
    print_text(f"Benchmark directory: {benchdir}")

    print_subtitle("Loading data...")
    latent_data = np.load(os.path.join(benchdir, "latent_vecs.npz"))
    smats, shape = {}, None
    for key in latent_data.keys():
        samples = latent_data[key]
        assert samples.ndim == 3, (
            "Expect samples with shape (n_samples, n_subjects, latent_dim).")
        if shape is None:
            shape = samples.shape
        n_samples, n_subjects, _ = samples.shape
        assert n_samples == shape[0], (
            "All samples must have the same number of samples.")
        assert n_subjects == shape[1], (
            "All samples must have the same number of subjects.")
        smats[key] = data2mat(samples)
        n_subjects = smats[key].shape[1]
        print_text(f"{key} similarities: {smats[key].shape}")
    meta_df = pd.read_csv(
        os.path.join(benchdir, f"latent_meta_{dataset}.tsv"), sep="\t")
    clinical_scores = ["asd", "age", "sex", "site", "fsiq"]
    scale_scores = ["ordinal", "ratio", "ordinal", "ratio", "ratio"]
    scores = dict((qname, scale)
                  for qname, scale in zip(clinical_scores, scale_scores))
    indices = range(n_subjects)
    cmats = dict()
    cidxs = dict()
    le = LabelEncoder()
    clinical_scores = meta_df.columns
    for qname in clinical_scores:
        if qname not in scores:
            print_error(f"Unknown score {qname}, use default ratio scale.")
        if qname in ("site", "sex"):
            meta_df[qname] = le.fit_transform(meta_df[qname].values)
        scale = scores.get(qname, "ratio")
        vec = meta_df[qname].values[indices]
        idx = ~np.isnan(vec)
        vec = vec[idx]
        cmat = vec2mat(vec, data_scale=scale)
        cmats[qname] = cmat
        cidxs[qname] = idx
        print_text(f"{qname} number of outliers measures: {np.sum(~idx)}")
        print_text(f"{qname} features similarities: {cmat.shape}")

    print_subtitle("Compute RSA...")
    data = dict((key, arr[:, indices][..., indices])
                for key, arr in smats.items())
    rsa_results, rsa_records = dict(), dict()
    for qname in clinical_scores:
        for key, smat in data.items():
            res = fit_rsa(smat, cmats[qname], idxs=cidxs[qname])
            n_samples = len(res)
            rsa_records.setdefault(key, []).extend(res.tolist())
            rsa_results.setdefault(qname, {})[key] = res
        rsa_records.setdefault("score", []).extend([qname] * n_samples)
    rsa_df = pd.DataFrame.from_dict(rsa_records)
    print(rsa_df.groupby("score").describe().loc[
        :, (slice(None), ["count", "mean", "std"])])
    rsa_df.to_csv(os.path.join(benchdir, "rsa.tsv"), sep="\t", index=False)

    print_subtitle("Display subject's (dis)similarity matrices...")
    ncols = n_samples
    nrows = len(data)
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    idx1 = 0
    for name, sdata in data.items():
        _name = " ".join(name.split("_")[:-1])
        for idx2, smat in enumerate(sdata):
            ax = plt.subplot(nrows, ncols, idx1 + 1)
            plot_mat(f"{_name} ({idx2 + 1})", smat, ax=ax, figsize=None,
                     dpi=300, fontsize=12)
            idx1 += 1
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{dataset.upper()} SUBJECTS (S) MAT", fontsize=20, y=.95)
    filename = os.path.join(benchdir, f"sub_mat_{dataset}.png")
    plt.savefig(filename)
    print_result(f"subjects mat: {filename}")

    print_subtitle("Display score's (dis)similarity matrices...")
    ncols = 4
    nrows = int(np.ceil(len(cmats) / ncols))
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    for idx, (name, cmat) in enumerate(cmats.items()):
        _name = " ".join(name.split("_"))
        ax = plt.subplot(nrows, ncols, idx + 1)
        plot_mat(_name.upper(), cmat, ax=ax, figsize=None, dpi=300,
                 fontsize=12)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{dataset.upper()} CLINICAL (C) MAT", fontsize=20, y=.95)
    filename = os.path.join(benchdir, f"clinical_mat_{dataset}.png")
    plt.savefig(filename)
    print_result(f"clinical mat: {filename}")

    print_subtitle("Display Kendall tau statistics...")
    ncols = 3
    nrows = int(np.ceil(len(clinical_scores) / ncols))
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    pairwise_stats = []
    for idx, qname in enumerate(clinical_scores):
        ax = plt.subplot(nrows, ncols, idx + 1)
        pairwise_stat_df = plot_bar(
            qname, rsa_results, ax=ax, figsize=None, dpi=300, fontsize=7,
            fontsize_star=12, fontweight="bold", line_width=2.5,
            marker_size=3, title=qname.upper(), report_t=True,
            do_one_sample_stars=True, do_pairwise_stars=True, palette="Set2")
        if pairwise_stat_df is not None:
            pairwise_stats.append(pairwise_stat_df)
    if len(pairwise_stats) > 0:
        pairwise_stat_df = pd.concat(pairwise_stats)
        pairwise_stat_df.to_csv(
            os.path.join(benchdir, "rsa_pairwise_stats.tsv"), sep="\t",
            index=False)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{dataset.upper()} RSA RESULTS", fontsize=20, y=.95)
    filename = os.path.join(benchdir, f"rsa_{dataset}.png")
    plt.savefig(filename)
    print_result(f"RSA: {filename}")
