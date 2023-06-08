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
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import torch
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_full_data
from mmbench.workflow.predict import get_predictor


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
    data_tr, meta_tr, data, meta = _data[0:4]
    for mod in modalities:
        data[mod] = data[mod].to(device).float()
        data_tr[mod] = data_tr[mod].to(device).float()
    meta_df, meta_df_tr = {}, {} 
    meta_df["asd"] = meta["asd"]
    meta_df_tr["asd"] = meta_tr["asd"]
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
    for i in range(n_samples):
        models.append(linear_model.LogisticRegression())
        models[i].fit(data_tr,meta_tr) # train_test_split ?
        print(models[i]) 

    print_subtitle("Evaluate models...")
    results = {}
    results_tr = {}
    res, res_cv= [], []
    print_text("model: logistic regression")
    _, scorer, name = get_predictor(meta_df_tr["asd"])
    for model in models:
        scores = cross_val_score(model, data_tr, meta_df_tr["asd"], cv=5, scoring=scorer, n_jobs=-1)
        res_cv.append(f"{scores.mean():.2f} +/- {scores.std():.2f}")
        res.append(scorer(model, data, meta_df["asd"]))
    res_cv_df = pd.DataFrame.from_dict(
                {"model": range(n_samples), "score": res_cv})
    res_cv_df["qname"] = "asd"
    print(res_cv_df)
    predict_results = {"asd": {"LogisticReg_ROI_euaims": np.asarray(res)}}
    predict_df = pd.DataFrame.from_dict(predict_results, orient="index")
    predict_df = pd.concat([predict_df[col].explode() for col in predict_df],
                           axis="columns")
    predict_df.to_csv(os.path.join(benchdir, "predict.tsv"), sep="\t",
                      index=False)
    _df = pd.concat(res_cv_df)
    _df.to_csv(os.path.join(benchdir, "predict_cv.tsv"), sep="\t",
               index=False)
