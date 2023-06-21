# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the workflows to generate embeddigns.
"""

# Imports
import os
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from mmbench.config import ConfigParser
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import (
    get_test_data, get_train_data, get_test_full_data, get_train_full_data)
from mmbench.workflow.predict import get_predictor
from mmbench.model import get_models, eval_models


def benchmark_latent_exp(dataset, datasetdir, configfile, outdir,
                         dtype="full", missing_modalities=None):
    """ Retrieve the learned latent space of different models using a
    N samplings scheme.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    configfile: str
        the path to the config file descibing the different models to compare.
        This configuration file is a Python (\*.py) file with a dictionary
        named '_models' containing the different model settings. Keys of this
        dictionary are the model names, each beeing described with a model
        getter function 'get' and associated kwargs 'get_kwargs', as weel as
        an evaluation function 'eval' and associated kwargs 'eval_kwargs'.
        The getter and evaluation functions are defined in the 'mmbench.model'
        module.
    outdir: str
        the destination folder.
    dtype: str, default 'full'
        the data type: 'complete' or 'full'.
    missing_modalities: list, default None
        remove data from missing modalities.

    Notes
    -----
    We need to extend this procedure to CV models.
    """
    print_title(f"GET MODELS LATENT VARIABLES: {dataset}")
    assert dtype in ("complete", "full")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")
    missing_modalities = missing_modalities or []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    if dtype == "full":
        train_loader, test_loader = (get_train_full_data, get_test_full_data)
    else:
        train_loader, test_loader = (get_train_data, get_test_data)
    data_test, meta_test_df = test_loader(dataset, datasetdir, modalities)
    data_train, meta_train_df = train_loader(dataset, datasetdir, modalities)
    for mod in modalities:
        data_test[mod] = data_test[mod].to(device).float()
        data_train[mod] = data_train[mod].to(device).float()
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

    print_subtitle("Parsing config...")
    parser = ConfigParser("latent-config", configfile)
    pprint(parser.config.models)

    print_subtitle("Loading models...")
    models = {}
    default_params = {
        "n_channels": len(modalities),
        "n_feats": [data_test[mod].shape[1] for mod in modalities],
        "n_feats_tr": [data_train[mod].shape[1] for mod in modalities],
        "modalities": modalities}
    for mod in missing_modalities:
        data_test[mod] = None
        data_train[mod] = None
    for name, params in parser.config.models.items():
        _models = get_models(
            params["get"],
            **parser.set_auto_params(params["get_kwargs"], default_params))
        eval_kwargs = parser.set_auto_params(
            params["eval_kwargs"], default_params)
        models[name] = (_models, params["eval"], eval_kwargs)
    for name, (_models, _, _) in models.items():
        print_text(f"model: {name}")
        print(_models[0])

    print_subtitle("Evaluate models...")
    test_results = {}
    train_results = {}
    for name, (_models, eval_fct, kwargs_fct) in models.items():
        print_text(f"model: {name}")
        for idx, model in enumerate(_models):
            if isinstance(model, torch.nn.Module):
                model = model.to(device)
                model.eval()
                _models[idx] = model
        with torch.set_grad_enabled(False):
            print_text("split: test")
            test_embeddings = eval_models(eval_fct, _models, data_test,
                                          **kwargs_fct)
            print_text("split: train")
            train_embeddings = eval_models(eval_fct, _models, data_train,
                                           **kwargs_fct)
            for key, val in test_embeddings.items():
                key = _sanitize(key)
                test_results[f"{key}_{dataset}"] = val
            for key, val in train_embeddings.items():
                key = _sanitize(key)
                train_results[f"{key}_{dataset}"] = val
    features_test_file = os.path.join(benchdir,
                                      f"latent_vecs_test_{dataset}.npz")
    features_train_file = os.path.join(benchdir,
                                       f"latent_vecs_train_{dataset}.npz")
    np.savez_compressed(features_test_file, **test_results)
    np.savez_compressed(features_train_file, **train_results)
    print_result(f"train features: {features_train_file}")
    print_result(f"test features: {features_test_file}")


def _sanitize(key):
    """ Sanitize the experiment name.
    """
    key = key.replace("rois", "ROI")
    key = key.replace("joint", "Joint")
    key = key.replace("clinical", "eCRF")
    return key
