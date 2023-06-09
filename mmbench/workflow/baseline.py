# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the baseline model.
"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import torch
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_full_data
from mmbench.workflow.predict import get_predictor
from mmbench.plotting import plot_bar


def benchmark_baseline(datasetdir, outdir, n_samples=10):
    """ Train and test a baseline model on euaims

    Parameters
    ----------
    datasetdir: str
        the path to the euaims associated data.
    outdir: str
        the destination folder.
    n_samples: int, default 10
        the number of models trained.
    """
    dataset = "euaims"
    print_title(f"GET MODELS LATENT VARIABLES: {dataset}")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    _data = get_full_data(dataset, datasetdir, modalities)
    data_tr, meta_df_tr, data, meta_df = _data[0:4]
    for mod in modalities:
        data[mod] = data[mod].to(device).float()
        data_tr[mod] = data_tr[mod].to(device).float()
    meta_df = meta_df[['asd']]
    meta_df_tr = meta_df_tr[['asd']]
    print_text([(key, arr.shape) for key, arr in data.items()])
    print_text(meta_df)
    print_text([(key, arr.shape) for key, arr in data_tr.items()])
    print_text(meta_df_tr)
    meta_file = os.path.join(benchdir, f"latent_meta_{dataset}.tsv")
    meta_file_tr = os.path.join(benchdir, f"latent_meta_train_{dataset}.tsv")
    meta_df.to_csv(meta_file, sep="\t", index=False)
    meta_df_tr.to_csv(meta_file_tr, sep="\t", index=False)
    print_result(f"metadata: {meta_file}")

    print_subtitle("Training models...")
    models = []
    samples = data_tr["rois"].cpu()
    y = meta_df_tr["asd"]
    qname = "asd"
    for i in range(n_samples):
        models.append(linear_model.LogisticRegression())
        models[i].fit(samples,y) # train_test_split ?
        print(models[i]) 

    print_subtitle("Evaluate models...")
    results = {}
    results_tr = {}
    res, res_cv= [], []
    print_text("model: logistic regression")
    _, scorer, name = get_predictor(y)
    for model in tqdm(models):
        scores = cross_val_score(model, samples, y, cv=5, scoring=scorer, n_jobs=-1)
        res_cv.append(f"{scores.mean():.2f} +/- {scores.std():.2f}")
        res.append(scorer(model, data["rois"].cpu(), meta_df))
    res_cv_df = pd.DataFrame.from_dict(
                {"model": range(n_samples), "score": res_cv})
    res_cv_df["qname"] = "asd"
    print(res_cv_df)
    predict_results = {"asd": {"LogisticReg_ROI_euaims": np.asarray(res)}}
    predict_df = pd.DataFrame.from_dict(predict_results, orient="index")
    predict_df = pd.concat([predict_df[col].explode() for col in predict_df],
                           axis="columns")
    predict_df.to_csv(os.path.join(benchdir, "baseline.tsv"), sep="\t",
                      index=False)
    _df = pd.concat([res_cv_df])
    _df.to_csv(os.path.join(benchdir, "baseline_cv.tsv"), sep="\t",
               index=False)

    print_subtitle("Display statistics...")
    ncols = 1
    nrows = 1
    plt.figure(figsize=np.array((ncols, nrows)) * 4)
    pairwise_stats = []
    ax = plt.subplot(nrows, ncols, 1)
    pairwise_stat_df = plot_bar(
        qname, predict_results, ax=ax, figsize=None, dpi=300, fontsize=7,
        fontsize_star=12, fontweight="bold", line_width=2.5,
        marker_size=3, title=qname.upper(),
        do_one_sample_stars=False, palette="Set2", yname=name)
    if pairwise_stat_df is not None:
        pairwise_stats.append(pairwise_stat_df)
    if len(pairwise_stats) > 0:
        pairwise_stat_df = pd.concat(pairwise_stats)
        pairwise_stat_df.to_csv(
            os.path.join(benchdir, "predict_pairwise_stats.tsv"), sep="\t",
            index=False)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{dataset.upper()} BASELINE RESULTS", fontsize=20, y=.95)
    filename = os.path.join(benchdir, f"baseline_{dataset}.png")
    plt.savefig(filename)
    print_result(f"BASELINE: {filename}")