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


def get_models(get_fct, checkpointfile, *args, **kwargs):
    """ Get N trained instance of a model from associated checkpoint files.

    Parameters
    ----------
    get_fct: callable
        a fonction that returns an instance of a model from a checkpoint file
        and some additional optional parameters.
    checkpointfile: list of str
        list of files containing the model weights.

    Returns
    -------
    models: list of Module
        the list of instanciated models.
    """
    if not isinstance(checkpointfile, (list, tuple)):
        checkpointfile = [checkpointfile]
    return [get_fct(path, *args, **kwargs) for path in checkpointfile]


def eval_models(eval_func, models, data, modalities, **kwargs):
    """ Evaluate N instance of a model.

    Parameters
    ----------
    eval_func: callable
        the model evaluation function.
    models: list of Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        name of the views to consider.
    kwargs: dict
        optional arguments passed to the evaluation function.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = None
    for model in models:
        emb = eval_func(model, data, modalities, **kwargs)
        if embeddings is None:
            embeddings = dict((key, [val]) for key, val in emb.items())
        else:
            assert sorted(embeddings.keys()) == sorted(emb.keys())
            for key, val in emb.items():
                embeddings[key].append(val)
    return embeddings


def get_mopoe(checkpointfile):
    """ Return the MOPOE model.

    Parameters
    ----------
    checkpointfile: str
        path to the model weights.

    Returns
    -------
    model: Module
        instanciated model.
    """
    flags_file = os.path.join(os.path.dirname(checkpointfile), os.pardir,
                              os.pardir, "flags.rar")
    if not os.path.isfile(flags_file):
        raise ValueError(f"Can't locate expermiental flags: {flags_file}.")
    alphabet_file = os.path.join(
        os.path.dirname(mopoe.__file__), "alphabet.json")
    print_text(f"restoring weights: {checkpointfile}")
    experiment, flags = MultimodalExperiment.get_experiment(
        flags_file, alphabet_file, checkpointfile)
    return experiment.mm_vae


def eval_mopoe(models, data, modalities, n_samples=10, verbose=1):
    """ Evaluate the MOPOE model.

    Parameters
    ----------
    models: Module
        input model.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    n_samples: int, default 10
        the number of time to sample the posterior.
    verbose: int, default 1
        control the verbosity level.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    inf_data = models.inference(data)
    latents = [inf_data["modalities"][f"{mod}_style"] for mod in modalities]
    latents += [inf_data["joint"]]
    key = "MoPoe"
    for idx, name in enumerate(modalities + ["joint"]):
        z_mu, z_logvar = latents[idx]
        if z_mu is None:
            key = "MoPoeClav"
            continue
        q = Normal(loc=z_mu, scale=torch.exp(0.5 * z_logvar))
        if n_samples == 1:
            z_samples = q.loc
        else:
            z_samples = q.sample((n_samples, ))
        code = z_samples.cpu().detach().numpy()
        if verbose > 0:
            print_text(f"{key}_{name} latents: {code.shape}")
        embeddings[f"{key}_{name}"] = code
    return embeddings


def get_smcvae(checkpointfile, n_channels, n_feats, **kwargs):
    """ Return the sMCVAE model.

    Parameters
    ----------
    checkpointfile: str
        path to the model weights.
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
        instanciated model.
    """
    model = MCVAE(n_channels=n_channels, n_feats=n_feats, sparse=True,
                  **kwargs)
    checkpoint = torch.load(checkpointfile, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    return model


def eval_smcvae(model, data, modalities, n_samples=10, threshold=0.2,
                ndim=None, verbose=1):
    """ Evaluate the sMCVAE model.

    Parameters
    ----------
    model: Module or list of Module
        input model.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    n_samples: int, default 10
        the number of time to sample the posterior.
    threshold: float, default 0.2
        value for thresholding. If None, no thresholding is applied.
    ndim: int, default None
        number of dimensions to keep.
    verbose: int, default 1
        control the verbosity level.

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
    z_samples = [z.reshape(-1, model.latent_dim) for z in z_samples]
    if threshold is not None:
        dim = [elem.ndim for elem in z_samples]
        z_samples = model.apply_threshold(
            z_samples, threshold=threshold, ndim=ndim, keep_dims=False,
            reorder=True)
        if [elem.ndim for elem in z_samples] != dim:
            z_samples = [elem.reshape(-1, 1) for elem in z_samples]
    thres_latent_dim = z_samples[0].shape[1]
    if n_samples > 1:
        z_samples = [z.reshape(n_samples, -1, thres_latent_dim)
                     for z in z_samples]
    for idx, name in enumerate(modalities):
        code = z_samples[idx]
        embeddings[f"sMCVAE_{name}"] = code
        if verbose > 0:
            print_text(f"sMCVAE_{name} latents: {code.shape}")
    return embeddings


def get_pls(checkpointfile):
    """ Return PLS models.

    Parameters
    ----------
    checkpointfile: str
        path to the model weights.

    Returns
    -------
    models: Module
        instanciated model.
    """
    model = load(checkpointfile)
    return model


def eval_pls(model, data, modalities, n_samples=1, verbose=1):
    """ Evaluate the PLS model.

    Parameters
    ----------
    model: Module
        input models.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    n_samples: int, default 1
        the number of time to sample the posterior.
    verbose: int, default 1
        control the verbosity level.

    Returns
    -------
    embeddings: dict
        the generated latent representations.
    """
    embeddings = {}
    Y_test, X_test = [data[mod].to(torch.float32) for mod in modalities]
    X_test_r = model.transform(
        X_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy())
    for idx, name in enumerate(modalities):
        code = np.array(X_test_r[idx])
        if verbose > 0:
            print_text(f"PLS_{name} latents: {code.shape}")
        embeddings[f"PLS_{name}"] = code
    return embeddings


def get_neuroclav(checkpointfile, layers=(444, 256, 20), **kwargs):
    """ Return the NeuroCLAV model.

    Parameters
    ----------
    checkpointfiles: str
        path to the model weights.
    layers: a list of int, default (444, 256, 20)
        the MLP layers definition.
    kwargs: dict
        extra parameters passed to the NeuroCLAV constructor.

    Returns
    -------
    model: Module
        the instanciated model.
    """
    from models.mlp import MLP
    model = MLP(layers=layers)
    checkpoint = torch.load(checkpointfile, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint)
    return model


def eval_neuroclav(model, data, modalities, n_samples=1, verbose=1):
    """ Evaluate the NeuroCLAV model.

    Parameters
    ----------
    model: Module
        the input model.
    data: dict
        the input data organized by views.
    modalities: list of str
        names of the model input views.
    verbose: int, default 1
        control the verbosity level.

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
        code = model(view_data).cpu().detach().numpy()
    code = np.array(code)
    if verbose > 0:
        print_text(f"NeuroCLAV_rois latents: {code.shape}")
    embeddings["NeuroCLAV_rois"] = code
    return embeddings
