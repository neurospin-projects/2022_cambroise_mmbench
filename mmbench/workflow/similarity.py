# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define a feature similarity experiment usinf the Centered Kernel Alignment
(CKA) as a measure of similarity between two output features in a layer.
"""

# Imports
import os
import copy
from pprint import pprint
import numpy as np
import pandas as pd
import torch
from mmbench.config import ConfigParser
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_test_data, get_full_data
from mmbench.model import get_models, eval_models
from brainboard.metric import linear_cka, layer_at, get_named_layers


def benchmark_feature_similarity_exp(dataset, datasetdir, configfile, outdir, transfer=False):
    """ Define the Centered Kernel Alignment (CKA) as a measure of similarity
    between two output features in a layer of a network architecture given
    any two pairs of instances of a network.

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
    transfer: bool, default False
        Training dataset is different from test dataset
    """
    print_title(f"COMPARE MODEL LATENT REPRESENTATIONS: {dataset}")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    data_test, meta_test_df = get_test_data(dataset, datasetdir, modalities)
    if transfer:
        _, _, _, _, data_test, meta_test_df = get_full_data(dataset, datasetdir, modalities)
    for mod in modalities:
        data_test[mod] = data_test[mod].to(device).float()
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
        if (not isinstance(checkpoints, (list, tuple))
                or "layers" not in params):
            continue
        _models = get_models(
            params["get"],
            **parser.set_auto_params(params["get_kwargs"], default_params))
        eval_kwargs = parser.set_auto_params(
            params["eval_kwargs"], default_params)
        models[name] = (_models, params["eval"], eval_kwargs, params["layers"])
    for name, (_models, _, _, _) in models.items():
        print_text(f"model: {name}")
        print(get_named_layers(_models[0]).keys())
        print(_models[0])

    print_subtitle("Evaluate models...")
    results_test = {}
    for name, (_models, eval_fct, eval_kwargs, layers) in models.items():
        if not isinstance(_models[0], torch.nn.Module):
            continue
        print_text(f"model: {name}")
        scores_test = {}
        for layer_name in layers:
            n_models = len(_models)
            iu = np.array(np.triu_indices(n_models, k=1)).T
            mat = np.zeros((n_models, n_models))
            _layer_data_test = []
            for model in _models:
                model = model.to(device)
                model.eval()
                with torch.set_grad_enabled(False):
                    _data, _ = layer_at(
                        model, layer_name, data_test,
                        eval_fct=eval_fct, eval_kwargs=eval_kwargs)
                    _layer_data_test.append(_data)
            for i1, i2 in iu:
                mat[i1, i2] = linear_cka(
                    _layer_data_test[i1], _layer_data_test[i2])
            mat += mat.T
            print(mat)
            scores_test[layer_name] = mat
        for layer_name in scores_test:
            results_test[f"{name}_{layer_name}"] = scores_test

    similarity_file = os.path.join(benchdir, f"cka_similarity_{dataset}.npz")
    np.savez_compressed(similarity_file, **results_test)
    print_result(f"CKA similarity: {similarity_file}")
