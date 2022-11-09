# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the different datasets.
"""

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from mopoe.multimodal_cohort.dataset import MultimodalDataset, DataManager
from mopoe.multimodal_cohort.dataset import MissingModalitySampler


def get_train_data(dataset, datasetdir, modalities):
    """ See `get_data` for documentation.
    """
    return get_data(dataset, datasetdir, modalities, dtype="train")


def get_test_data(dataset, datasetdir, modalities):
    """ See `get_data` for documentation.
    """
    return get_data(dataset, datasetdir, modalities, dtype="test")


def get_data(dataset, datasetdir, modalities, dtype):
    """ Load the train/test data.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    modalities: list of str
        the modalities to load.
    dtype: str
        the data type: 'train' or 'test'.

    Returns
    -------
    data: dict
        the loaded data for each modality.
    metadata: DataFrame
        the associated meta information.
    """
    trainset, testset = get_dataset(dataset, datasetdir, modalities)
    if dtype == "train":
        dataset = trainset
    else:
        dataset = testset
    sampler = MissingModalitySampler(dataset, batch_size=len(dataset))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    dataiter = iter(loader)
    while True:
        data, _, meta = dataiter.next()
        if all([mod in data.keys() for mod in modalities]):
            break
    scores = data["clinical"].T
    clinical_names = np.load(
        os.path.join(datasetdir, "clinical_names.npy"), allow_pickle=True)
    clinical_names = [name.replace("t1_", "") for name in clinical_names]
    meta = dict((key, val.numpy() if isinstance(val, torch.Tensor) else val)
                for key, val in meta.items())
    del meta["participant_id"]
    meta.update(dict((key, val) for key, val in zip(clinical_names, scores)))
    meta_df = pd.DataFrame.from_dict(meta)
    return data, meta_df


def get_dataset(dataset, datasetdir, modalities):
    """ Load the train/test datasets.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    modalities: list of str
        the modalities to load.

    Returns
    -------
    trainset, testset: MultimodalDataset
        the loaded train/test datasets.
    """
    manager = DataManager(
        dataset, datasetdir, modalities, overwrite=False,
        allow_missing_blocks=False)
    scalers = set_scalers(manager.train_dataset, modalities)
    transform = {
        mod: transforms.Compose([
            unsqueeze_0,
            scaler.transform,
            transforms.ToTensor(),
            torch.squeeze]) for mod, scaler in scalers.items()}
    trainset = MultimodalDataset(
        manager.fetcher.train_input_path,
        manager.fetcher.train_metadata_path,
        on_the_fly_transform=transform)
    testset = MultimodalDataset(
        manager.fetcher.test_input_path,
        manager.fetcher.test_metadata_path,
        on_the_fly_transform=transform)
    return trainset, testset


def set_scalers(dataset, modalities):
    """ Apply a standard scaler modality by modality.

    Parameters
    ----------
    dataset: MultimodalDataset
        a multi modal dataset.
    modalities: list of str
        the modalities to load.

    Returns
    -------
    scalers: dict
        a fitted standard scaler for each modality.
    """
    all_data = {}
    for data, label, meta in dataset:
        for mod in modalities:
            if mod in data.keys():
                all_data.setdefault(mod, []).append(data[mod])
    scalers = {}
    for mod in modalities:
        scaler = StandardScaler()
        scaler.fit(all_data[mod])
        scalers[mod] = scaler
    return scalers


def unsqueeze_0(x):
    """ Returns a new tensor with a dimension of size one at dimension 0.
    """
    return x.unsqueeze(0)
