# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the prediction workflows.
"""
# Imports
import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
from mmbench.dataset import get_train_data, get_test_data
from sklearn.cross_decomposition import PLSRegression
from mmbench.color_utils import print_title, print_subtitle
from joblib import dump


def benchmark_pls_exp(
        dataset, datasetdir, outdir, fit_lat_dims=10):
    """ Train the PLS model

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    fit_lat_dims: int, default 10
        the number of latent dimensions.

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
    X_test, _ = get_test_data(dataset, datasetdir, modalities)
    del X_test["index"]
    print("test:", [(key, arr.shape) for key, arr in X_test.items()])
    Y_train, X_train = [X_train[mod].to(torch.float32) for mod in modalities]
    Y_test, X_test = [X_test[mod].to(torch.float32) for mod in modalities]

    print_subtitle("Create model...")
    pls = PLSRegression(n_components=fit_lat_dims)
    pls.fit(X_train, Y_train)

    print_subtitle("Save model...")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    model_file = os.path.join(outdir, "pls_model.joblib")
    dump(pls, model_file)
