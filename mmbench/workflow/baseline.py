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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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

    print_subtitle("Loading parameters...")
    params_file = pd.read_csv(os.path.join(
        benchdir, f"supervised-baseline_metric-BAcc_{dataset}.tsv"), sep="\t")
    imax = params_file["mean_test_score"].idxmax()
    parameters = {
        "penalty": params_file["param_logistic__penalty"][imax],
        "C": params_file["param_logistic__C"][imax],
        "residualize": params_file["param_residualizer__scale"][imax],
        "rois_mod": params_file["param_selector__modalities"][imax]}
    print("parameters:", parameters)

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    data_train, meta_train_df = get_train_full_data(
        dataset, datasetdir, modalities, residualize=False)
    data_test, meta_test_df = get_test_full_data(
        dataset, datasetdir, modalities, residualize=False)
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
    meta_test_df.to_csv(meta_test_file, sep="\t", index=True)
    meta_train_df.to_csv(meta_train_file, sep="\t", index=True)
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
    print(f"train: {X_train.shape} - {y_train.shape}")
    print(f"test: {X_test.shape} - {y_test.shape}")
    logreg = LogisticRegression(penalty=parameters["penalty"],
                                C=parameters["C"])
    logreg.fit(X_train, y_train)
    print("Theoretical accuracy:", params_file["mean_test_score"][imax])
    print("Accuracy obtained:", logreg.score(X_test, y_test))
    print("Accuracy on train:", logreg.score(X_train, y_train))
