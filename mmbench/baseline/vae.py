# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the VAE model.
"""

# Imports
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from mmbench.dataset import get_train_data, get_test_data
from mmbench.color_utils import print_title, print_subtitle, print_result
from mmbench.workflow.smcvae import train_model


def train_vae(dataset, datasetdir, outdir, fit_lat_dims=20, n_iter=100,
              host="http://localhost", port=8085):
    """ Train the VAE model.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.
    fit_lat_dims: int, default 20
        the number of latent dimensions.
    n_iter: int, default 100
        the number of train iterations.
    host: str, default 'http://localhost'
        the host on which visdom is launched.
    port: int, default 8085
        the port on which the visdom server is launched.
    """
    from brainboard import Board
    from brainite.models import VAE
    from brainite.losses import BetaHLoss

    print_title("Training VAE...")
    print_subtitle("Generating data loader...")
    modalities = ["rois", "clinical"]
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    X_train, _ = get_train_data(dataset, datasetdir, modalities)
    del X_train["index"]
    X_test, _ = get_test_data(dataset, datasetdir, modalities)
    del X_test["index"]
    X_train = znorm(X_train["rois"]).to(torch.float32)
    X_test = znorm(X_test["rois"]).to(torch.float32)
    print("- train:", X_train.shape)
    print("- test:", X_test.shape)
    datasets = {
        "train": torch.utils.data.TensorDataset(X_train),
        "val": torch.utils.data.TensorDataset(X_test)}
    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split], batch_size=100,
            shuffle=(True if split == "train" else False), num_workers=1)
        for split in ["train", "val"]}

    print_subtitle("Generating model...")
    model_name = "vae"
    model = VAE(input_channels=1, input_dim=X_train.shape[1],
                conv_flts=None, dense_hidden_dims=[256],
                latent_dim=fit_lat_dims, noise_fixed=True, dropout=0.1,
                sparse=False)
    print(model)

    print_subtitle("Fitting model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001,
                                 weight_decay=0.01)
    criterion = BetaHLoss(beta=10, steps_anneal=0, use_mse=True)
    scheduler = MyStepLR(optimizer=optimizer, init_lr=0.001, min_lr=0.0001,
                         step_size=100, gamma=0.9)
    board = Board(host=host, env=f"{dataset}_{model_name}", port=port)
    train_model(dataloaders, model, device, criterion, optimizer,
                scheduler=scheduler, board=board, n_epochs=n_iter)
    path = os.path.join(outdir, f"{model.__class__.__name__}_final.pth")
    torch.save(model.state_dict(), path)


class MyStepLR(torch.optim.lr_scheduler.MultiStepLR):
    """ Custom step LR scheduler.
    """
    def __init__(self, optimizer, init_lr, min_lr, step_size, gamma):
        n_steps = int(np.log(min_lr / init_lr) / np.log(gamma))
        milestones = [step_size * idx for idx in range(1, n_steps)]
        super(MyStepLR, self).__init__(
            optimizer, milestones=milestones, gamma=gamma)

    def step(self, metric=None, epoch=None):
        super(MyStepLR, self).step()


def znorm(arr):
    print(arr)
    std, mean, eps = (1., 0., 1e-8)
    return std * (arr - torch.mean(arr)) / (torch.std(arr) + eps) + mean
