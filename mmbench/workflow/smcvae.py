# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Train the sparse Multi-Channels Variational Auto-Encoder (sMCVAE).
"""

# Imports
import os
import sys
import time
import copy
import progressbar
import numpy as np
import torch
from torch.utils.data import TensorDataset
from mmbench.dataset import get_train_data, get_test_data
from mmbench.color_utils import print_title


def train_smcvae(dataset, datasetdir, outdir, fit_lat_dims=10, beta=1,
                 adam_lr=2e-3, n_epochs=10000, host="http://localhost",
                 port=8085):
    """ Train the sparse Multi-Channels Variational Auto-Encoder (sMCVAE).

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    fit_lat_dims: int, default 10
        the number of latent dimensions.
    beta: float, default 1
        the loss beta-VAE weight (0.5 for HBN).
    adam_lr: float, default 2e-3
        the initial learning rate in the ADAM optimizer.
    n_epochs: int, default 10000
        the number of training epochs.
    host: str, default 'http://localhost'
        the host on which visdom is launched.
    port: int, default 8085
        the port on which the visdom server is launched.
    """
    from brainboard import Board
    from brainite.models import MCVAE
    from brainite.losses import MCVAELoss

    print_title("Load dataset...")
    modalities = ["clinical", "rois"]
    X_train, _ = get_train_data(dataset, datasetdir, modalities)
    # train_indices = X_train["index"]
    del X_train["index"]
    print("train:", [(key, arr.shape) for key, arr in X_train.items()])
    X_test, _ = get_test_data(dataset, datasetdir, modalities)
    # test_indices = X_test["index"]
    del X_test["index"]
    print("test:", [(key, arr.shape) for key, arr in X_test.items()])

    print_title("Create data loaders...")
    X_train = [X_train[mod].to(torch.float32) for mod in modalities]
    X_test = [X_test[mod].to(torch.float32) for mod in modalities]
    print("train:", [arr.shape for arr in X_train])
    datasets = {
        "train": TensorDataset(*X_train),
        "val": TensorDataset(*X_test)}
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split], batch_size=len(datasets[split]),
            shuffle=(True if split == "train" else False), num_workers=1)
        for split in ["train", "val"]}

    print_title("Create model...")
    model_name = "smcvae"
    n_channels = len(X_train)
    n_feats = [X.shape[1] for X in X_train]
    checkpointdir = os.path.join(outdir, "checkpoints")
    if not os.path.isdir(checkpointdir):
        os.mkdir(checkpointdir)
    model = MCVAE(
        latent_dim=fit_lat_dims, n_channels=n_channels, n_feats=n_feats,
        vae_model="dense", vae_kwargs={}, sparse=True, noise_init_logvar=-3,
        noise_fixed=False)
    print(f" model: {model_name}")
    print(model)
    board = Board(host=host, env=f"{dataset}_{model_name}", port=port)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
    criterion = MCVAELoss(n_channels, beta=beta, sparse=True)

    print_title("Train model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(dataloaders, model, device, criterion, optimizer,
                n_epochs=(n_epochs + 1), board=board,
                checkpointdir=checkpointdir, board_updates=update_dropout_rate,
                save_after_epochs=100)


def train_model(dataloaders, model, device, criterion, optimizer,
                scheduler=None, n_epochs=100, checkpointdir=None,
                save_after_epochs=1, board=None, board_updates=None,
                load_best=False):
    """ General function to train a model and display training metrics.

    Parameters
    ----------
    dataloaders: dict of torch.utils.data.DataLoader
        the train & validation data loaders.
    model: nn.Module
        the model to be trained.
    device: torch.device
        the device to work on.
    criterion: torch.nn._Loss
        the criterion to be optimized.
    optimizer: torch.optim.Optimizer
        the optimizer.
    scheduler: torch.optim.lr_scheduler, default None
        the scheduler.
    n_epochs: int, default 100
        the number of epochs.
    checkpointdir: str, default None
        a destination folder where intermediate models/histories will be
        saved.
    save_after_epochs: int, default 1
        determines when the model is saved and represents the number of
        epochs before saving.
    board: brainboard.Board, default None
        a board to display live results.
    board_updates: list of callable, default None
        update displayed item on the board.
    load_best: bool, default False
        optionally load the best model regarding the loss.
    """
    since = time.time()
    if board_updates is not None:
        board_updates = listify(board_updates)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max
    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}
    model = model.to(device)
    with progressbar.ProgressBar(max_value=n_epochs) as bar:
        for epoch in range(n_epochs):
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_extra_loss = {}
                for batch_data in dataloaders[phase]:
                    if isinstance(batch_data, list):
                        batch_data = batch_data[0]
                    batch_data = to_device(batch_data, device)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward:
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs, layer_outputs = model(batch_data)
                        criterion.layer_outputs = layer_outputs
                        try:
                            loss, extra_loss = criterion(outputs)
                        except:

                            loss, extra_loss = criterion(outputs, batch_data)
                        for key, val in extra_loss.items():
                            if key not in running_extra_loss:
                                running_extra_loss[key] = val.item()
                            else:
                                running_extra_loss[key] += val.item()
                        # Backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    # Statistics
                    running_loss += loss.item() * batch_data[0].size(0)
                    for key in running_extra_loss.keys():
                        running_extra_loss[key] *= batch_data[0].size(0)
                if scheduler is not None and phase == "train":
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_extra_loss = copy.deepcopy(running_extra_loss)
                for key in epoch_extra_loss.keys():
                    epoch_extra_loss[key] /= dataset_sizes[phase]
                if board is not None:
                    if epoch % 25 == 0:
                        board.update_plot(
                            "loss_{0}".format(phase), epoch, epoch_loss)
                        for name, val in epoch_extra_loss.items():
                            board.update_plot(
                                "{0}_{1}".format(name, phase), epoch, val)
                # Display validation classification results
                if (board is not None and board_updates is not None and
                        phase == "val"):
                    if epoch % 25 == 0:
                        for update in board_updates:
                            update(model, board, outputs, layer_outputs)
                # Deep copy the best model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            # Save intermediate results
            if checkpointdir is not None and epoch % save_after_epochs == 0:
                outfile = os.path.join(
                    checkpointdir, "model_{0}.pth".format(epoch))
                checkpoint(
                    model=model, outfile=outfile, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch, epoch_loss=epoch_loss)
            bar.update(epoch)
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:4f}".format(best_loss))
    # Load best model weights
    if load_best:
        model.load_state_dict(best_model_wts)


def listify(data):
    """ Ensure that the input is a list or tuple.

    Parameters
    ----------
    arr: list or array
        the input data.

    Returns
    -------
    out: list
        the liftify input data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return data
    else:
        return [data]


def to_device(data, device):
    """ Transfer data to device.

    Parameters
    ----------
    data: tensor or list of tensor
        the data to be transfered.
    device: torch.device
        the device to work on.

    Returns
    -------
    out: tensor or list of tensor
        the transfered data.
    """
    if isinstance(data, list):
        return [tensor.to(device) for tensor in data]
    else:
        return data.to(device)


def checkpoint(model, outfile, optimizer=None, scheduler=None,
               **kwargs):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: nn.Module
        the model to be saved.
    outfile: str
        the destination file name.
    optimizer: torch.optim.Optimizer
        the optimizer.
    scheduler: torch.optim.lr_scheduler, default None
        the scheduler.
    kwargs: dict
        others parameters to be saved.
    """
    kwargs.update(model=model.state_dict())
    if optimizer is not None:
        kwargs.update(optimizer=optimizer.state_dict())
    if scheduler is not None:
        kwargs.update(scheduler=scheduler.state_dict())
    torch.save(kwargs, outfile)


def update_dropout_rate(model, board, outputs, layer_outputs=None):
    """ Display the dropout rate.
    """
    if model.log_alpha is not None:
        do = np.sort(model.dropout.numpy().reshape(-1))
        board.update_hist("dropout_probability", do)
