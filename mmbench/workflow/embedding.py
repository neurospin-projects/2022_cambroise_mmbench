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
import torch
from mmbench.config import ConfigParser
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_test_data
from mmbench.model import (get_mopoe, get_smcvae, eval_mopoe, eval_smcvae,
                           get_pls, eval_pls)


def benchmark_latent_exp(dataset, datasetdir, configfile, outdir):
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

    Notes
    -----
    We need to extend this procedure to CV models.
    """
    print_title(f"GET MODELS LATENT VARIABLES: {dataset}")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_subtitle("Loading data...")
    modalities = ["clinical", "rois"]
    print_text(f"modalities: {modalities}")
    data, meta_df = get_test_data(dataset, datasetdir, modalities)
    for mod in modalities:
        data[mod] = data[mod].to(device).float()
    print_text([(key, arr.shape) for key, arr in data.items()])
    print_text(meta_df)
    meta_file = os.path.join(benchdir, f"latent_meta_{dataset}.tsv")
    meta_df.to_csv(meta_file, sep="\t", index=False)
    print_result(f"metadata: {meta_file}")

    print_subtitle("Parsing config...")
    parser = ConfigParser("latent-config", configfile)
    pprint(parser.config.models)

    print_subtitle("Loading models...")
    models = {}
    default_params = {
        "n_channels": len(modalities),
        "n_feats": [data[mod].shape[1] for mod in modalities],
        "modalities": modalities}
    for name, params in parser.config.models.items():
        model = params["get"](
            **parser.set_auto_params(params["get_kwargs"], default_params))
        eval_kwargs = parser.set_auto_params(
            params["eval_kwargs"], default_params)
        models[name] = (model, params["eval"], eval_kwargs)
    for name, (model, _, _) in models.items():
        print_text(f"model: {name}")
        print(model)

    print_subtitle("Evaluate models...")
    results = {}
    for name, (model, eval_fct, kwargs_fct) in models.items():
        print_text(f"model: {name}")
        if not(name == "PLS"):
            model = model.to(device)
            model.eval()
        with torch.set_grad_enabled(False):
            embeddings = eval_fct(model, data, **kwargs_fct)
            for key, val in embeddings.items():
                key = _sanitize(key)
                results[f"{key}_{dataset}"] = val
    features_file = os.path.join(benchdir, f"latent_vecs_{dataset}.npz")
    np.savez_compressed(features_file, **results)
    print_result(f"features: {features_file}")


def _sanitize(key):
    """ Sanitize the experiment name.
    """
    key = key.replace("rois", "ROI")
    key = key.replace("joint", "Joint")
    key = key.replace("clinical", "eCRF")
    return key
