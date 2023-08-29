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
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_train_full_data, get_test_full_data
from mmbench.workflow.predict import get_predictor


def benchmark_baseline(datasetdir, outdir, solver='roc_auc'):
    """ Train and test a baseline model on euaims

    Parameters
    ----------
    datasetdir: str
        the path to the euaims associated data.
    outdir: str
        the destination folder.
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
        dataset, datasetdir, modalities, residualize=False)
    data_test, meta_test_df = get_test_full_data(
        dataset, datasetdir, modalities, residualize=False)
    # meta_test_df["asd"] = meta_test_df["asd"].apply(lambda x: x - 1)
    # meta_train_df["asd"] = meta_train_df["asd"].apply(lambda x: x - 1)
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
    print(f"train: {X_train.shape} - {y_train.shape}")
    print(f"test: {X_test.shape} - {y_test.shape}")
    ridge = linear_model.RidgeClassifier()
    logreg = CalibratedClassifierCV(ridge, method='isotonic')
    parameters = {
        "estimator__alpha": np.logspace(-2, 4, 7),
        "estimator__solver": ["auto", "lsqr", "sparse_cg", "saga"]}
    if solver == 'BAcc':
        logreg = LogisticRegression(max_iter=400, solver="saga", l1_ratio=0.5)
        parameters = {
            "penalty": ["l2", "l1", "elasticnet"],
            "C": np.logspace(-4, 1, 6)}
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=42)
    l_indices = msss.split(
        list(meta_train_df.index), meta_train_df.values)
    clf = GridSearchCV(logreg, parameters, cv=l_indices, scoring="roc_auc_ovr",
                       return_train_score=True, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Tuned Hyperparameters:", clf.best_params_)
    print("Accuracy (CV validation):", clf.best_score_)
    logreg = clf.best_estimator_
    score = logreg.score(X_test, y_test)
    print("Accuracy (test):", score)

    print_subtitle("Saving result...")
    res = {}
    res['std_test_score'] = clf.cv_results_['std_test_score']
    res['mean_test_score'] = clf.cv_results_['mean_test_score']
    res['params'] = clf.cv_results_['params']
    res['std_test_score'] = clf.cv_results_['std_train_score']
    res['best_score'] = [score] * len(res['params'])
    res_df = pd.DataFrame.from_dict(res)
    res_df.to_csv(os.path.join(outdir, "baseline.tsv"), sep="\t",
                  index=False)
    print_result("PREDICT: baseline.tsv")
