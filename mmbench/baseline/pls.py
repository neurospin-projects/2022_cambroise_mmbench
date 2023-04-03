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
import torch
from mmbench.dataset import get_train_data
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from mmbench.color_utils import print_title, print_subtitle
from joblib import dump


def train_pls(dataset, datasetdir, outdir, fit_lat_dims=3, n_samples=10,
              random_state=None):
    """ Train the PLS model

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    fit_lat_dims: int, default 3
        the number of latent dimensions.
    n_samples: int, default 10
        the number of models generated
    random_state: list of int, default None
        controls the shuffling applied to the data before applying the split.
        Pass a list of n_sampoles int for reproducible output across multiple
        function calls.

    Note
    ----
    The generated model is stored in 'outdir' in a file named
    'pls_model.joblib'. 'outdir' must correspond to the path given in the
    configuration file for the PLS checkpointfile.
    """
    print_title(" PLS ")
    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    X_train, _ = get_train_data(dataset, datasetdir, modalities)
    del X_train["index"]
    print("train:", [(key, arr.shape) for key, arr in X_train.items()])
    Y_train, X_train = [X_train[mod].to(torch.float32) for mod in modalities]
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    print_subtitle("Create models...")
    if random_state is None:
        random_state = [None] * n_samples
    for idx in range(n_samples):
        Xi_train, _, Yi_train, _ = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=random_state[idx])
        pls = PLSRegression(n_components=fit_lat_dims)
        pls.fit(Xi_train, Yi_train)
        model_file = os.path.join(outdir, f"pls_model_{idx}.joblib")
        dump(pls, model_file)
