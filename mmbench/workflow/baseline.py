# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the workflows to generate the baseline ASD supervised predictions on
EUAIMS.
"""

# Imports
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_train_full_data, get_test_full_data
from mmbench.workflow.predict import get_predictor
from mmbench.plotting import plot_bar


def benchmark_baseline(datasetdir, outdir, n_iter=10, random_state=None):
    """ Train and test a baseline model on euaims

    Parameters
    ----------
    datasetdir: str
        the path to the euaims associated data.
    outdir: str
        the destination folder.
    n_iter: int, default 10
        the number of trained models using different train set partitioning.
    random_state: int, default None
        controls the shuffling applied to the data before applying the split.
    """
    dataset = "euaims"
    print_title(f"GET MODELS LATENT VARIABLES: {dataset}")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    data_train, meta_train_df = get_train_full_data(
        dataset, datasetdir, modalities, residualize=True)
    data_test, meta_test_df = get_test_full_data(
        dataset, datasetdir, modalities, residualize=True)
    meta_test_df["asd"] = meta_test_df["asd"].apply(lambda x: x - 1)
    meta_train_df["asd"] = meta_train_df["asd"].apply(lambda x: x - 1)
    print_text([(key, arr.shape) for key, arr in data_test.items()])
    print_text(meta_test_df)
    print_text([(key, arr.shape) for key, arr in data_train.items()])
    print_text(meta_train_df)
    meta_test_file = os.path.join(
        benchdir, f"latent_meta_test_{dataset}.tsv")
    meta_train_file = os.path.join(
        benchdir, f"latent_meta_train_{dataset}.tsv")
    meta_test_df.to_csv(meta_test_file, sep="\t", index=False)
    meta_train_df.to_csv(meta_train_file, sep="\t", index=False)
    print_result(f"train metadata: {meta_train_file}")
    print_result(f"test metadata: {meta_test_file}")

    print_subtitle("Training a classification model...")
    print_text("model: LogisticRegression")
    X_train, X_test = (data_train["rois"].numpy(), data_test["rois"].numpy())
    y_train, y_test = (meta_train_df.asd.values, meta_test_df.asd.values)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    X_train = X_train
    X_test = X_test
    # ToDo: test random
    # y_train = shuffle(y_train)
    print(f"train: {X_train.shape} - {y_train.shape}")
    print(f"test: {X_test.shape} - {y_test.shape}")
    if random_state is None:
        random_states = [None] * n_iter
    else:
        random_states = [random_state + idx for idx in range(n_iter)]
    models = []
    cv_data = []
    for idx in range(n_iter):
        print_text(f"-> train model: {idx +  1}/{n_iter}")
        # ToDo: use iterative stratifier
        Xi_train, Xi_val, yi_train, yi_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_states[idx],
            stratify=y_train)
        print(f"distribution: {Counter(yi_train)}")
        model = LogisticRegression(max_iter=150)
        model.fit(Xi_train, yi_train)
        cv_data.append((Xi_train, yi_train))
        models.append(model)
        print(f"train score: {model.score(Xi_train, yi_train)}")
        print(f"val score: {model.score(Xi_val, yi_val)}")

    print_subtitle("Evaluate trained models...")
    train_metrics, test_metrics = [], []
    _, scorer, name = get_predictor(y_train)
    print_text(f"metric: {name}")
    for idx, model in tqdm(enumerate(models)):
        Xi_train, yi_train = cv_data[idx]
        test_metrics.append(scorer(model, X_test, y_test))
        train_metrics.append(scorer(model, Xi_train, yi_train))
    print(f"train score: {np.mean(train_metrics)} +/- {np.std(train_metrics)}")
    print(f"test score: {np.mean(test_metrics)} +/- {np.std(test_metrics)}")
    metric_df = pd.DataFrame.from_dict({"model": range(1, n_iter + 1),
                                        f"train_{name}": train_metrics,
                                        f"test_{name}": test_metrics})
    filename = os.path.join(
        benchdir, f"supervised-baseline_{name}_{dataset}.tsv")
    metric_df.to_csv(filename, sep="\t", index=False)
    print_result(f"supervised baseline: {filename}")

    print_subtitle("Display statistics...")
    predict_results = {
        "ASD": {"LogisticReg_ROI_train_euaims": np.asarray(train_metrics),
                "LogisticReg_ROI_test_euaims": np.asarray(test_metrics)}}
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plot_bar(
        "ASD", predict_results, ax=ax, figsize=None, dpi=300, fontsize=7,
        fontsize_star=12, fontweight="bold", line_width=2.5,
        marker_size=3, title=None, do_one_sample_stars=False, palette="Set2",
        yname=name)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)
    plt.suptitle(f"{dataset.upper()} SUPERVISED BASELINE", fontsize=18, y=1.)
    filename = os.path.join(
        benchdir, f"supervised-baseline_{name}_{dataset}.png")
    plt.savefig(filename)
    print_result(f"supervised baseline: {filename}")
