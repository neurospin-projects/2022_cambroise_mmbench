# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Train a constrastive Variational Auto-Encoder (cVAE).
"""

# Imports
import os
import sys
import time
import copy
import progressbar
import torch
from torch.utils.data import TensorDataset
from mmbench.dataset import EUAIMSContrastiveDataset
from mmbench.color_utils import print_title
from cvae.model import mmcVAE
from cvae.loss import mmcVAELoss


def train_cvae(dataset, datasetdir, outdir, general_lat_dims=15,
               specific_lat_dims=5, beta=4, lambda1=1, lambda2=2,
               adam_lr=1e-4, n_epochs=1000, host="http://localhost",
               port=8085):
    """ Train a contrastive Variational Auto-Encoder (cVAE).

    Parameters
    ----------
    dataset: str
        the dataset name: euaims.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    general_lat_dims: int, default 15
        the number of latent dimensions in the general part of the latent
        space.
    specific_lat_dims: int, default 5
        the number of latent dimensions in the specific part of the latent
        space.
    beta: float, default 4
        weight of the KL divergence.
    lambda1: float, default 1
        weight for the salient disentanglement loss.
    lambda2: float, default 2
        weight for the background disentanglement loss.
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

    print_title("Load dataset...")
    train_dataset = EUAIMSContrastiveDataset(datasetdir, train=True)
    scaler = train_dataset.scaler
    test_dataset = EUAIMSContrastiveDataset(
        datasetdir, train=False, scaler=scaler)
    datasets = {
        "train": train_dataset,
        "val": test_dataset}

    print_title("Create data loaders...")
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split], batch_size=len(datasets[split]),
            shuffle=(True if split == "train" else False), num_workers=1)
        for split in ["train", "val"]}
    dataiter = iter(dataloaders["train"])
    data, _ = dataiter.next()

    print_title("Create model...")
    model_name = "cvae"
    checkpointdir = os.path.join(outdir, "checkpoints")
    if not os.path.isdir(checkpointdir):
        os.mkdir(checkpointdir)
    model = mmcVAE(
        input_channels=1, input_dim=data.shape[-2:], conv_flts=None,
        dense_hidden_dims=[128],
        latent_dims=[general_lat_dims, specific_lat_dims])
    print(f" model: {model_name}")
    print(model)
    board = Board(host=host, env=f"{dataset}_{model_name}", port=port)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
    criterion = mmcVAELoss(beta=beta, lambda1=lambda1, lambda2=lambda2,
                           use_mse=True)

    print_title("Train model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(dataloaders, model, device, criterion, optimizer,
                n_epochs=(n_epochs + 1), board=board,
                checkpointdir=checkpointdir, save_after_epochs=100)


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
                    dataloaders[phase].dataset.reset_mapping()
                else:
                    model.eval()
                running_loss = 0.0
                running_extra_loss = {}
                for batch_data in dataloaders[phase]:
                    batch_data = to_device(batch_data, device)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward:
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs, layer_outputs = model(batch_data)
                        criterion.layer_outputs = layer_outputs
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
