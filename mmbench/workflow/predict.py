# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the predicction workflows.
"""
# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result,
    print_error)
from mmbench.plotting import plot_bar

def benchmark_pred_exp(dataset, datasetdir, outdir):
    """ Compare the learned latent space of different models using
    prediction analysis.

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
    print_title(f"COMPARE MODELS USING REGRESSIONS "
                "OR CLASSIFICATION WITH ML ANALYSIS: {dataset}")
    benchdir = outdir
    print_text(f"Benchmark directory: {benchdir}")

    print_subtitle("Loading data...")
    
    latent_data = np.load(os.path.join(benchdir, f"latent_vecs_{dataset}.npz"))
    latent_data_tr = np.load(os.path.join(benchdir,
                                          f"latent_vecs_train_{dataset}.npz"))
    assert latent_data.keys() == latent_data_tr.keys(), (
            "latent data must have the same keys")
    shape, shape_samples = None, None
    meta_df = pd.read_csv(
        os.path.join(benchdir, f"latent_meta_{dataset}.tsv"), sep="\t")
    meta_df_tr = pd.read_csv(
        os.path.join(benchdir, f"latent_meta_train_{dataset}.tsv"), sep="\t")
    clinical_scores = meta_df_tr.columns
    reg = linear_model.Ridge(alpha=.5)
    cla = linear_model.RidgeClassifier()
    predict_records, predict_results = dict(), dict()
    for key in latent_data.keys():
        samples = latent_data_tr[key]
        samples_test = latent_data[key]
        assert samples.ndim == 3 and samples_test.ndim == 3, (
            "Expect samples with shape (n_samples, n_subjects, latent_dim).")
        n_samples, n_subjects, _ = samples.shape
        _, n_subjects_test, _ = samples_test.shape
        if shape is None:
            shape = samples.shape + (n_subjects_test,)
        assert n_samples == shape[0] and samples_test.shape[0] == shape[0], (
            "All samples must have the same number of samples.")
        assert n_subjects == shape[1], (
            "All samples must have the same number of subjects.")
        assert n_subjects_test == shape[3], (
            "All samples must have the same number of subjects for testing.")
    res = [None]*n_samples
    for qname in clinical_scores:
        y_tr = meta_df_tr[qname]
        y = meta_df[qname]
        for key in latent_data.keys():
            samples = latent_data_tr[key]
            samples_test = latent_data[key]
            if detect_cla(y):
                for i in range(n_samples):
                    cla.fit(samples[i], y_tr)
                    res[i] = cla.score(samples_test[i], y)
            else:
                for i in range(n_samples):
                    reg.fit(samples[i], y_tr)
                    res[i] = reg.score(samples_test[i], y)
            predict_records.setdefault(key, []).extend(res)
            predict_results.setdefault(qname, {})[key] = np.asarray(res)
        predict_records.setdefault("score", []).extend([qname]* n_samples)
    predict_df = pd.DataFrame.from_dict(predict_records)
    predict_df.to_csv(os.path.join(benchdir, "predict.tsv"), sep="\t",
                      index=False)

    print_subtitle("Display statistics...")
    ncols = 3
    nrows = int(np.ceil(len(clinical_scores) / ncols))
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    pairwise_stats = []
    for idx, qname in enumerate(clinical_scores):
        ax = plt.subplot(nrows, ncols, idx + 1)
        pairwise_stat_df = plot_bar(
            qname, predict_results, ax=ax, figsize=None, dpi=300, fontsize=7,
            fontsize_star=12, fontweight="bold", line_width=2.5,
            marker_size=3, title=qname.upper(),
            do_one_sample_stars=False, palette="Set2")
        if pairwise_stat_df is not None:
            pairwise_stats.append(pairwise_stat_df)
    if len(pairwise_stats) > 0:
        pairwise_stat_df = pd.concat(pairwise_stats)
        pairwise_stat_df.to_csv(
            os.path.join(benchdir, "predict_pairwise_stats.tsv"), sep="\t",
            index=False)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{dataset.upper()} PREDICT RESULTS", fontsize=20, y=.95)
    filename = os.path.join(benchdir, f"predict_{dataset}.png")
    plt.savefig(filename)
    print_result(f"PREDICT: {filename}")


def detect_cla(data):
    cla = False
    err = 0
    if isinstance(data[0], (str, int)):
        return True
    for e in data:
        err = err + e - int(e)
    if err == 0:
        return True
    return False
