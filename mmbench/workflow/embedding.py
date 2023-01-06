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
import numpy as np
import pandas as pd
import torch
from mmbench.color_utils import (
    print_title, print_subtitle, print_text, print_result)
from mmbench.dataset import get_test_data
from mmbench.model import get_mopoe, get_smcvae, eval_mopoe, eval_smcvae


def benchmark_latent_exp(dataset, datasetdir, outdir, smcvae_checkpointfile,
                         mopoe_checkpointfile, smcvae_kwargs=None):
    """ Retrieve the learned latent space of different models using a
    10 samplings scheme.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    run: str
        the name of the experiment in the destination folder
        `<dataset>_<timestamp>`.
    smcvae_checkpointfile: str
        the path to the sMCVAE model weights.
    mopoe_checkpointfile: str
        the path to the MOPOE model weights.
    smcvae_kwargs: dict, default None
        optionally give extra parameters to construct the model.

    Notes
    -----
    We need to extend this procedure to CV models.
    """
    print_title(f"GET MODELS LATENT VARIABLES: {dataset}")
    benchdir = outdir
    if not os.path.isdir(benchdir):
        os.mkdir(benchdir)
    print_text(f"Benchmark directory: {benchdir}")
    smcvae_kwargs = smcvae_kwargs or {}
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

    print_subtitle("Loading models...")
    models = {}
    # > MoPoe
    models["MoPoe"] = (get_mopoe(mopoe_checkpointfile), eval_mopoe,
                       {"modalities": modalities})
    # > sMCVAE
    n_channels = len(modalities)
    n_feats = [data[mod].shape[1] for mod in modalities[::-1]]
    models["sMCVAE"] = (
        get_smcvae(smcvae_checkpointfile, n_channels, n_feats,
                   **smcvae_kwargs),
        eval_smcvae, {"modalities": modalities[::-1]})
    for name, (model, _, _) in models.items():
        print_text(f"model: {name}")
        print(model)

    print_subtitle("Evaluate models...")
    results = {}
    n_samples = 10
    for name, (model, eval_fct, kwargs_fct) in models.items():
        print_text(f"model: {name}")
        model = model.to(device)
        model.eval()
        with torch.set_grad_enabled(False):
            embeddings = eval_fct(model, data, n_samples=n_samples,
                                  **kwargs_fct)
            for key, val in embeddings.items():
                key = _sanitize(key)
                results[f"{key}_{dataset}"] = val
    features_file = os.path.join(benchdir, "latent_vecs.npz")
    np.savez_compressed(features_file, **results)
    print_result(f"features: {features_file}")


def _sanitize(key):
    """ Sanitize the experiment name.
    """
    key = key.replace("rois", "ROI")
    key = key.replace("joint", "Joint")
    key = key.replace("clinical", "eCRF")
    return key
