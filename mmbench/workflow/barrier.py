# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define barriere experiments.
"""

# Imports
import os
import copy
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
import torch
from mmbench.config import ConfigParser
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_test_data, get_train_data
from mmbench.workflow.predict import get_predictor
from brainboard.metric import eval_interpolation


def benchmark_barrier_exp(dataset, datasetdir, configfile, outdir,
                          downstream_name):
    """ Compare the performance barrier interpolating the weights of any two
    pairs of intances of the same network and monitoring a common downstream
    task.

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
    downstream_name: str
        the name of the column that contains the downstream classification
        task.
    """
    print_title(f"COMPARE MODEL WEIGHTS: {dataset}")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    data_train, meta_train_df = get_train_data(dataset, datasetdir, modalities)
    data_test, meta_test_df = get_test_data(dataset, datasetdir, modalities)
    y_train = meta_train_df[downstream_name]
    y_test = meta_test_df[downstream_name]
    for mod in modalities:
        data_train[mod] = data_train[mod].to(device).float()
        data_test[mod] = data_test[mod].to(device).float()
    print_text([(key, arr.shape) for key, arr in data_train.items()])
    print_text(meta_train_df)
    print_text([(key, arr.shape) for key, arr in data_test.items()])
    print_text(meta_test_df)

    print_subtitle("Parsing config...")
    parser = ConfigParser("latent-config", configfile)
    pprint(parser.config.models)

    print_subtitle("Loading models...")
    models = {}
    default_params = {
        "n_channels": len(modalities),
        "n_feats": [data_test[mod].shape[1] for mod in modalities],
        "modalities": modalities}
    for name, params in parser.config.models.items():
        checkpoints = params["get_kwargs"]["checkpointfile"]
        if not isinstance(checkpoints, list):
            continue
        _models = []
        for path in checkpoints:
            _params = copy.deepcopy(params)
            _params["get_kwargs"]["checkpointfile"] = path
            model = params["get"](
                **parser.set_auto_params(_params["get_kwargs"],
                                         default_params))
            _models.append(model)
        eval_kwargs = parser.set_auto_params(
            params["eval_kwargs"], default_params)
        eval_kwargs["n_samples"] = 1
        models[name] = (_models, params["eval"], eval_kwargs)
    for name, (_models, _, _) in models.items():
        print_text(f"model: {name}")
        print(_models[0])

    print_subtitle("Evaluate models...")

    def eval_fn(model, loaders, y_train, y_test, eval_fn=None,
                eval_kwargs=None):
        model.eval()
        with torch.no_grad():
            X = []
            for data in loaders:
                if eval_fn is not None:
                    z = eval_fn(model, data, **eval_kwargs).values()
                    z = np.concatenate(list(z), axis=1)
                else:
                    z = model(data).cpu().detach().numpy()
                X.append(z)
        X_train, X_test = X
        clf, scoring = get_predictor(y_train)
        clf.fit(X_train, y_train)
        scorer = skmetrics.get_scorer(scoring)
        return scorer(clf, X_test, y_test)

    results_test = {}
    for name, (_models, eval_fct, eval_kwargs) in models.items():
        print_text(f"model: {name}")
        kwargs = {"eval_fn": eval_fct, "eval_kwargs": eval_kwargs,
                  "y_train": y_train, "y_test": y_test}
        n_models = len(_models)
        iu = np.array(np.triu_indices(n_models, k=0)).T
        mat = np.zeros((n_models, n_models))
        for i1, i2 in iu:
            model1 = _models[i1].to(device).eval()
            model2 = _models[i2].to(device).eval()
            state1 = model1.state_dict()
            state2 = model2.state_dict()
            coeffs, metrics = eval_interpolation(
                model1, state1, state2, [data_train, data_test], eval_fn,
                n_coeffs=10, eval_kwargs=kwargs)
            print(coeffs)
            print(metrics)
            mat[i1, i2] = np.trapz(metrics, coeffs)
            mat[i2, i1] = mat[i1, i2]
        print(mat)
        results_test[name] = mat

    barrier_file = os.path.join(benchdir, f"barrier_interp_{dataset}.npz")
    np.savez_compressed(barrier_file, **results_test)
    print_result(f"barrier interpolation: {barrier_file}")
