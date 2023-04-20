# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the NeuroClav weakly supervised model.
"""
# Imports
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from mmbench.dataset import get_train_data, get_test_data
from mmbench.color_utils import print_title, print_subtitle


def train_neuroclav(dataset, datasetdir, outdir, fit_lat_dims=20, n_iter=100):
    """ Train the NeuroClav model.

    Parameters
    ----------
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
    """
    from manifold.neuroclav import NeuroCLAV
    from data.collate import TwoViewsCollateFunction
    from data.augmentations.base import Compose, Transform
    from data.augmentations.intensity import ZNormalization

    class NeuroClavDataset(Dataset):
        """ Define a dataset that can be used with NeuroClav.
        """
        def __init__(self, X_train, y_train, X_test, y_test, phenotypes,
                     split="train", znorm=True):
            """ Init class.

            Parameters
            -----------
            X_train: ndarray (N, M)
                the input train data, where N is the number of subjets and M
                the number of ROIs.
            y_train: ndarray (N, K)
                the train phenotypes, where N is the number of subjects and K
                the number of auxiliary variables.
            X_test: ndarray (L, M)
                the input train data, where L is the number of subjets and M
                the number of ROIs.
            y_test: ndarray (L, K)
                the train phenotypes, where L is the number of subjects and K
                the number of auxiliary variables.
            phenotypes: list of str
                list of phenotype names.
            split: str, default 'train'
                split to instantiate in ('train', 'test').
            znorm: boolean, default True
                whether the phenotypes will be z-normalized across subjects or
                not.
            """
            assert split in ["train", "test"], f"Unknown split: {split}"
            for X, y in ((X_train, y_train), (X_test, y_test)):
                assert len(X) == len(y), (
                    "Different number of subjects in X and y.")
            if not isinstance(phenotypes, (list, tuple)):
                phenotypes = [phenotypes]
            for y in (y_train, y_test):
                assert y.shape[1] == len(phenotypes), (
                    "Different number of phenotypes in y and phenotypes.")
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            self.phenotypes = phenotypes
            self.split = split
            if znorm:
                scaler = StandardScaler().fit(self.y_train)
                self.y_train = scaler.transform(self.y_train)
                self.y_test = scaler.transform(self.y_test)
            if split == "train":
                self.X, self.y = self.X_train, self.y_train
            else:
                self.X, self.y = self.X_test, self.y_test
            self.n_subjects = len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

        def __len__(self):
            return self.n_subjects

    class RandomCutoutCollateFunction(TwoViewsCollateFunction):
        """ Implements the transformations for NeuroClav with random cutout.
        """
        def __init__(self, n_iter=10, value=0.0, znorm=True, p=0.5):
            """ Init class.

            Parameters
            ----------
            n_iter: int, default 10
               number of erased areas.
            value: float, default 0.0
                the replacement value.
            znorm: bool, default True
               whether the data will be z-normalized across subjects or not.
            p: float, default 0.5
               probability to apply the transformation.
            """
            transform = [RandomCutout((1, 1), n_iter, value, p=p)]
            if znorm:
                transform.append(ZNormalization())
            transform = Compose(transform)
            super(RandomCutoutCollateFunction, self).__init__(transform)

    class RandomCutout(Transform):
        """ Define a radom cutout transformation for ROI-like data.
        """
        def __init__(self, n_iter=1, value=0.0, inplace=False, **kwargs):
            super().__init__(**kwargs)
            self.n_iter = n_iter
            self.value = value
            self.inplace = inplace

        def parse_data(self, data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                raise ValueError("Unexpected type: %s" % type(data))

        def _erase_area(self, arr, idx):
            arr[idx] = self.value
            return arr

        def apply_transform(self, arr):
            if not self.inplace:
                arr = np.copy(arr)
            indices = np.random.choice(len(arr), self.n_iter, replace=False)
            for idx in indices:
                arr = self._erase_area(arr, idx)
            return arr

    print_title("Training NeuroClav...")
    print_subtitle("Generating data loader...")
    modalities = ["rois", "clinical"]
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    X_train, _ = get_train_data(dataset, datasetdir, modalities)
    del X_train["index"]
    X_test, _ = get_test_data(dataset, datasetdir, modalities)
    del X_test["index"]
    X_train, y_train = [X_train[mod].to(torch.float32) for mod in modalities]
    X_test, y_test = [X_test[mod].to(torch.float32) for mod in modalities]
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = clinical_names.tolist()
    print("- train:", X_train.shape, y_train.shape)
    print("- test:", X_test.shape, y_test.shape)
    print("- phenotypes:", clinical_names, len(clinical_names))
    datasets = dict(
        (split, NeuroClavDataset(X_train, y_train, X_test, y_test,
                                 clinical_names, split=split, znorm=True))
        for split in ("train", "test"))

    print_subtitle("Generating model...")
    model = NeuroCLAV(encoder=f"mlp{X_train.shape[1]}x256",
                      kernel="gaussian",
                      n_components=fit_lat_dims,
                      batch_size=128,
                      max_iteration=n_iter,
                      learning_rate_init=1e-4,
                      dir=outdir,
                      valid_frequency=1,
                      save_frequency=10,
                      validation_fraction=None,
                      num_workers=1)
    print(model)
    print(model._build_model(model.encoder))

    print_subtitle("Fitting model...")
    n_erase = int(X_train.shape[1] * 0.1)
    print("- n erase:", n_erase)
    collate_fn = RandomCutoutCollateFunction(n_iter=n_erase, p=0.8, znorm=True)
    model.fit(datasets["train"], collate_fn=collate_fn,
              aux_names=clinical_names)
    model.save()
    path = os.path.join(outdir, f"{model.model_.__class__.__name__}_final.pth")
    torch.save(model.model_.state_dict(), path)
