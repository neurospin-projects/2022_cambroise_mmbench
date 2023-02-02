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
import torch
from torch.utils.data import TensorDataset
from mmbench.dataset import EUAIMSDataset
from mmbench.color_utils import print_title
from cvae.model import mmcVAE
from cvae.loss import mmcVAELoss
from cvae.utils import train_model


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
    train_dataset = EUAIMSDataset(datasetdir, train=True)
    scaler = train_dataset.scaler
    test_dataset = EUAIMSDataset(datasetdir, train=False, scaler=scaler)
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
    data, _, _ = dataiter.next()

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
