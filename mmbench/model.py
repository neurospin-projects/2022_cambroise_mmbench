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
        list of paths to model weights.

    Returns
    -------
    model: Module
        instanciated models.
    """
    models = []
    if not isinstance(checkpointfile, (list, tuple)):
        checkpointfile = [checkpointfile]
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


def eval_mopoe(models, data, modalities, n_samples=10, _disp=True):
    """ Evaluate the MOPOE model.

    Parameters
    ----------
    models: Module or list of Module
        input models.
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
    if isinstance(models, list):
        embeddings = multi_eval(eval_mopoe, models, data, modalities,
                                n_samples=n_samples, _disp=False)
        for key in embeddings:
            print_text(f"{key} latents: {embeddings[key].shape}")
        return embeddings

    inf_data = models.inference(data)
    latents = [inf_data["modalities"][f"{mod}_style"] for mod in modalities]
    latents += [inf_data["joint"]]
    key = "MoPoe"
    for idx, name in enumerate(modalities + ["joint"]):
        z_mu, z_logvar = latents[idx]
        if z_mu is None:
            key = "MoPoeT"
            continue
        q = Normal(loc=z_mu, scale=torch.exp(0.5 * z_logvar))
        if n_samples == 1:
            z_samples = q.loc
            code = z_samples.cpu().detach().numpy()
        else:
            z_samples = q.sample((n_samples, ))
            code = z_samples.cpu().detach().numpy()
        if _disp:
            print_text(f"{key}_{name} latents: {code.shape}")
        embeddings[f"{key}_{name}"] = code
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
    if not isinstance(checkpointfile, (list, tuple)):
        checkpointfile = [checkpointfile]
    for model_file in checkpointfile:
        model = MCVAE(n_channels=n_channels, n_feats=n_feats, sparse=True,
                      **kwargs)
        checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])
        models.append(model)
    return models


def eval_smcvae(models, data, modalities, threshold=0.2, n_samples=10,
                ndim=None, _disp=True):
    """ Evaluate the sMCVAE model.

    Parameters
    ----------
    models: Module or list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    threshold: float, default 0.2
        value for thresholding
    n_samples: int, default 10
        the number of time to sample the posterior.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    if isinstance(models, list):
        embeddings = multi_eval(eval_smcvae, models, data, modalities,
                                threshold=threshold, n_samples=n_samples,
                                ndim=ndim, _disp=False)
        for key in embeddings:
            print_text(f"{key} latents: {embeddings[key].shape}")
        return embeddings

    latents = models.encode([data[mod] for mod in modalities])
    if n_samples == 1:
        z_samples = [q.loc.cpu().detach().numpy() for q in latents]
    else:
        z_samples = [q.sample((n_samples, )).cpu().detach().numpy()
                     for q in latents]
    z_samples = [z.reshape(-1, models.latent_dim) for z in z_samples]
    if threshold is not None:
        dim = []
        for elem in z_samples:
            dim.append(elem.ndim)
        z_samples = apply_threshold(models, z_samples, threshold=threshold,
                                    ndim=ndim, keep_dims=False, reorder=True)
        if [elem.ndim for elem in z_samples] != dim:
            z_samples = [elem.reshape(-1, 1) for elem in z_samples]
    thres_latent_dim = z_samples[0].shape[1]
    if n_samples > 1:
        z_samples = [z.reshape(n_samples, -1, thres_latent_dim)
                     for z in z_samples]
    for idx, name in enumerate(modalities):
        code = z_samples[idx]
        embeddings[f"sMCVAE_{name}"] = code
        if _disp:
            print_text(f"sMCVAE_{name} latents: {code.shape}")
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
    if not isinstance(checkpointfile, (list, tuple)):
        checkpointfile = [checkpointfile]
    for file in checkpointfile:
        models.append(load(file))
    return models


def eval_pls(models, data, modalities, n_samples=1, _disp=True):
    """ Evaluate the PLS model.

    Parameters
    ----------
    models: list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    n_samples: int, default 1
        the number of time to sample the posterior.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    if isinstance(models, list):
        embeddings = multi_eval(eval_pls, models, data, modalities,
                                n_samples=n_samples, _disp=False)
        for key in embeddings:
            print_text(f"{key} latents: {embeddings[key].shape}")
        return embeddings

    Y_test, X_test = [data[mod].to(torch.float32) for mod in modalities]
    X_test_r = models.transform(
        X_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy())
    for idx, name in enumerate(modalities):
        code = np.array(X_test_r[-idx - 1])
        if _disp:
            print_text(f"PLS_{name} latents: {code.shape}")
        embeddings[f"PLS_{name}"] = code
    return embeddings


def get_neuroclav(checkpointfile, layers, **kwargs):
    """ Return the NeuroCLAV model.

    Parameters
    ----------
    checkpointfiles: str
        the path to the model weights.
    layers: a list of int
        a parameter passed to the NeuroCLAV constructor.
    kwargs: dict
        extra parameters passed to the NeuroCLAV constructor.

    Returns
    -------
    model: Module
        the instanciated model.
    """
    from models.mlp import MLP
    models = []
    if not isinstance(checkpointfile, (list, tuple)):
        checkpointfile = [checkpointfile]
    for file in checkpointfile:
        model = MLP(layers=layers)
        checkpoint = torch.load(file, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint)
        models.append(model)
    return models


def eval_neuroclav(models, data, modalities, n_samples=1, _disp=True):
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
    if isinstance(models, list):
        embeddings = multi_eval(eval_neuroclav, models, data, modalities,
                                n_samples=n_samples, _disp=False)
        for key in embeddings:
            print_text(f"{key} latents: {embeddings[key].shape}")
        return embeddings

    assert "rois" in modalities  # TODO: use function parameters
    view_data = data["rois"]
    with torch.no_grad():
        z_samples = models(view_data)
    code = z_samples.cpu().detach().numpy()
    code = np.array(code)
    if _disp:
        print_text(f"NeuroCLAV_rois latents: {code.shape}")
    embeddings["NeuroCLAV_rois"] = code
    return embeddings


def multi_eval(eval_func, models, data, modalities, **kwargs):
    """ Evaluate a list of models.

    Parameters
    ----------
    eval_func: evaluation function
        evaluation function to call
    models: list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    kwargs: {n_samples, threshold}
        optional arguments of the evaluation functions

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    for model in models:
        emb = eval_func(model, data, modalities, **kwargs)
        for key in emb:
            if key not in embeddings:
                embeddings[key] = np.empty((0,) + emb[key].shape)
            embeddings[key] = np.append(embeddings[key], [emb[key]], axis=0)
    for key in embeddings:
        if embeddings[key].ndim == 4:
            shape = embeddings[key].shape
            embeddings[key] = embeddings[key].reshape(shape[0] * shape[1],
                                                      shape[2], shape[3])
    return embeddings


def apply_threshold(model, z, threshold, keep_dims=True, reorder=False,
                    ndim=None):
    """ Apply dropout threshold.

    Parameters
    ----------
    model: MCVAE
        input model
    z: Tensor
        distribution samples.
    threshold: float
        dropout threshold.
    keep_dims: bool default True
        dropout lower than threshold is set to 0.
    reorder: bool default False
        reorder dropout rates.
    ndim: int, default None
        number of dimensions to keep

    Returns
    -------
    z_keep: list
        dropout rates.
    """
    assert 0 < threshold <= 1.0, (
        f"the threshold ({threshold}) must be between 0 and 1")
    order = torch.argsort(model.dropout).squeeze()
    keep = (model.dropout < threshold).squeeze()
    if (ndim is not None and torch.sum(keep).item() != ndim):
        keep, threshold = create_keep(model, threshold, ndim)
    z_keep = []
    for drop in z:
        if keep_dims:
            drop[:, ~keep] = 0
        else:
            drop = drop[:, keep]
            order = torch.argsort(
                model.dropout[model.dropout < threshold]).squeeze()
        if reorder:
            drop = drop[:, order]
        z_keep.append(drop)
        del drop
    return z_keep


def create_keep(model, threshold, ndim):
    """ Create keep list with ndim selected distribution samples.

    Parameters
    ----------
    model: MCVAE
        input model
    threshold: float
        initial dropout threshold.
    ndim: int
        number of dimensions to keep

    Returns
    -------
    keep: list
        selected distribution samples.
    threshold: float
        final dropout threshold.
    """
    keep = (model.dropout < threshold).squeeze()
    n, tmin, tmax = 0, 0, 1
    while (torch.sum(keep).item() != ndim and n < 50):
        if torch.sum(keep).item() < ndim:
            tmin = threshold
            threshold = (threshold + tmax) / 2
        else:
            tmax = threshold
            threshold = (threshold + tmin) / 2
        keep = (model.dropout < threshold).squeeze()
        n = n + 1
    assert (n < 50)
    return keep, threshold
