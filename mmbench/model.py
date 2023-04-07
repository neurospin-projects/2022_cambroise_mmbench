# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the different models.
"""

# Imports
import os
import torch
import numpy as np
from brainite.models import MCVAE
import mopoe
from mopoe.multimodal_cohort.experiment import MultimodalExperiment
from mmbench.color_utils import print_text
from joblib import load


def get_mopoe(checkpointfile):
    """ Return the MOPOE model.

    Parameters
    ----------
    checkpointfile: str
        list of paths to model weights.

    Returns
    -------
    model: Module
        instanciated models.
    """
    models = []
    for model_file in checkpointfile:
        flags_file = os.path.join(os.path.dirname(model_file), os.pardir,
                                  os.pardir, "flags.rar")
        if not os.path.isfile(flags_file):
            raise ValueError(f"Can't locate expermiental flags: {flags_file}.")
        alphabet_file = os.path.join(
            os.path.dirname(mopoe.__file__), "alphabet.json")
        print_text(f"restoring weights: {model_file}")
        experiment, flags = MultimodalExperiment.get_experiment(
            flags_file, alphabet_file, model_file)
        models.append(experiment.mm_vae)
    return models


def eval_mopoe(models, data, modalities):
    """ Evaluate the MOPOE model.

    Parameters
    ----------
    models: Module or list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    z_mu = tuple([] for _ in range(len(modalities) + 1))
    if not isinstance(models, list):
        models = [models]
    for model in models:
        inf_data = model.inference(data)
        latents = [inf_data["modalities"][f"{mod}_style"]
                   for mod in modalities]
        latents += [inf_data["joint"]]
        for idx, name in enumerate(modalities + ["joint"]):
            z_mu[idx].append(latents[idx][0].cpu().detach().numpy())
    for idx, name in enumerate(modalities + ["joint"]):
        code = np.array(z_mu[idx])
        print_text(f"{name} latents: {code.shape}")
        embeddings[f"MoPoe_{name}"] = code
    return embeddings


def get_smcvae(checkpointfile, n_channels, n_feats, **kwargs):
    """ Return the sMCVAE model.

    Parameters
    ----------
    checkpointfile: str
        list of paths to model weights.
    latent_dim: int
        the number of latent dimensions.
    n_channels: int
        the number of input channels/views.
    n_feats: list of int
        the number of features for each channel.
    kwargs: dict
        extra parameters passed to the MCVAE constructor.

    Returns
    -------
    model: Module
        instanciated models.
    """
    models = []
    for model_file in checkpointfile:
        model = MCVAE(n_channels=n_channels, n_feats=n_feats, sparse=True,
                      **kwargs)
        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])
        models.append(model)
    return models


def eval_smcvae(models, data, modalities):
    """ Evaluate the sMCVAE model.

    Parameters
    ----------
    models: Module or list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    code = []
    if not isinstance(models, list):
        models = [models]
    for idx, model in enumerate(models):
        latents = model.encode([data[mod] for mod in modalities])
        z_samples = [q.sample((1, )).cpu().detach().numpy() for q in latents]
        z_samples = [z.reshape(-1, model.latent_dim) for z in z_samples]
        z_samples = model.apply_threshold(
            z_samples, threshold=0.2, keep_dims=False, reorder=True)
        thres_latent_dim = z_samples[0].shape[1]
        z_samples = [z.reshape(1, -1, thres_latent_dim) for z in z_samples]
        code.append([z_mod[0] for z_mod in z_samples])
    code = np.array(code)
    code = code.transpose((1, 0, 2, 3))
    for idx, name in enumerate(modalities):
        if code[idx].shape[0] == 1:
            embeddings[f"sMCVAE_{name}"] = code[idx][0]
        else:
            print_text(f"{name} latents: {code[idx].shape}")
            embeddings[f"sMCVAE_{name}"] = code[idx]
    return embeddings


def get_pls(checkpointfile):
    """ Return PLS models.

    Parameters
    ----------
    checkpointfile: list of str
        list of paths to model weights.

    Returns
    -------
    models: list of Module
        instanciated models.
    """
    models = []
    for file in checkpointfile:
        models.append(load(file))
    return models


def eval_pls(models, data, modalities):
    """ Evaluate the PLS model.

    Parameters
    ----------
    models: list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    Y_test, X_test = [data[mod].to(torch.float32) for mod in modalities]
    latent = ([], [])
    for model in models:
        X_test_r = model.transform(
            X_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy())
        latent[0].append(X_test_r[0])
        latent[1].append(X_test_r[1])
    for idx, name in enumerate(modalities):
        code = np.array(latent[idx])
        print_text(f"{name} latents: {code.shape}")
        embeddings[f"PLS_{name}"] = code
    return embeddings
