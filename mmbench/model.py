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
from joblib import load
from torch.distributions import Normal
from brainite.models import MCVAE
import mopoe
from mopoe.multimodal_cohort.experiment import MultimodalExperiment
from mmbench.color_utils import print_text


def get_mopoe(checkpointfile):
    """ Return the MOPOE model.

    Parameters
    ----------
    checkpointfile: str
        the path to the model weights.

    Returns
    -------
    model: Module
        the instanciated model.
    """
    flags_file = os.path.join(
        os.path.dirname(checkpointfile), os.pardir, os.pardir,
        "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError(f"Can't locate expermiental flags: {flags_file}.")
    alphabet_file = os.path.join(
        os.path.dirname(mopoe.__file__), "alphabet.json")
    print_text(f"restoring weights: {checkpointfile}")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, alphabet_file, checkpointfile)
    return experiment.mm_vae


def eval_mopoe(model, data, modalities, n_samples=10):
    """ Evaluate the MOPOE model.

    Parameters
    ----------
    model: Module
        the input model.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    n_samples: int, default 10
        the number of time to sample the posterior.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    inf_data = model.inference(data)
    latents = [inf_data["modalities"][f"{mod}_style"] for mod in modalities]
    latents += [inf_data["joint"]]
    for idx, name in enumerate(modalities + ["joint"]):
        z_mu, z_logvar = latents[idx]
        q = Normal(loc=z_mu, scale=torch.exp(0.5 * z_logvar))
        if n_samples == 1:
            z_samples = q.loc
        else:
            z_samples = q.sample((n_samples, ))
        code = z_samples.cpu().detach().numpy()
        print_text(f"{name} latents: {code.shape}")
        embeddings[f"MoPoe_{name}"] = code
    return embeddings


def get_smcvae(checkpointfile, n_channels, n_feats, **kwargs):
    """ Return the sMCVAE model.

    Parameters
    ----------
    checkpointfile: str
        the path to the model weights.
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
        the instanciated model.
    """
    model = MCVAE(n_channels=n_channels, n_feats=n_feats, sparse=True,
                  **kwargs)
    checkpoint = torch.load(checkpointfile, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    return model


def eval_smcvae(model, data, modalities, n_samples=10):
    """ Evaluate the sMCVAE model.

    Parameters
    ----------
    model: Module
        the input model.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    n_samples: int, default 10
        the number of time to sample the posterior.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    latents = model.encode([data[mod] for mod in modalities])
    if n_samples == 1:
        z_samples = [q.loc.cpu().detach().numpy() for q in latents]
    else:
        z_samples = [q.sample((n_samples, )).cpu().detach().numpy()
                     for q in latents]
    z_samples = [z.reshape(-1, model.latent_dim)
                 for z in z_samples]
    z_samples = model.apply_threshold(
        z_samples, threshold=0.2, keep_dims=False, reorder=True)
    thres_latent_dim = z_samples[0].shape[1]
    if n_samples > 1:
        z_samples = [z.reshape(n_samples, -1, thres_latent_dim)
                     for z in z_samples]
    for idx, name in enumerate(modalities):
        code = z_samples[idx]
        print_text(f"{name} latents: {code.shape}")
        embeddings[f"sMCVAE_{name}"] = code
    return embeddings


def get_pls(checkpointfile):
    """ Return PLS models.

    Parameters
    ----------
    checkpointfile: str
        the path to the model weights.

    Returns
    -------
    models: Module
        instanciated models.
    """
    models = []
    for path in checkpointfile:
        models.append(load(path))
    return models


def eval_pls(models, data, modalities):
    """ Evaluate the PLS model.

    Parameters
    ----------
    models: Module
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


def get_neuroclav(checkpointfile, n_feats, **kwargs):
    """ Return the NeuroCLAV model.

    Parameters
    ----------
    checkpointfile: str
        the path to the model weights.
    kwargs: dict
        extra parameters passed to the NeuroCLAV constructor.

    Returns
    -------
    model: Module
        the instanciated model.
    """
    from models.mlp import MLP

    model = MLP(layers=(444, 256, 20))  # TODO: use function parameters
    checkpoint = torch.load(checkpointfile, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    return model


def eval_neuroclav(model, data, modalities):
    """ Evaluate the NeuroCLAV model.

    Parameters
    ----------
    model: Module
        the input model.
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
    assert "rois" in modalities  # TODO: use function parameters
    view_data = data["rois"]
    model.eval()
    with torch.no_grad():
        embeddings = model(view_data)
    return embeddings.cpu().detach().numpy()
