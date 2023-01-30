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
from torch.utils.data import Dataset
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


class EUAIMSContrastiveDataset(Dataset):
    """ Create the EUAIMS contrastive dataset.
    """
    def __init__(self, datasetdir, train=True, transform=None,
                 flatten=False, seed=None, scaler=None):
        """ Init class.

        Parameters
        ----------
        datasetdir: str
            the path to the dataset associated data.
        train: bool, default True
            specifies training or test dataset.
        transform: callable, default None
            optional transform to be applied on a sample.
        flatten: bool, default False
            optionally select all subjects.
        seed: int, default None
            for reproducibility fix a seed.
        scaler: sklearn-like scaler, default None
            optionally set a fitted scaler.
        """
        if seed is not None:
            raise NotImplementedError("Setting a seed is not supported yet.")

        split = "train" if train else "test"
        meta_split_file = os.path.join(datasetdir, f"metadata_{split}.tsv")
        df = pd.read_csv(meta_split_file, sep="\t")
        subjects = df["participant_id"].values
        roi_file = os.path.join(datasetdir, "rois_data.npy")
        data = np.load(roi_file)
        all_subjects = np.load(roi_file.replace("_data.npy", "_subjects.npy"))
        indices = np.nonzero(np.in1d(all_subjects, subjects))[0]
        print(data.shape, len(subjects), len(indices))
        data = data[indices]
        self.scaler = scaler or StandardScaler()
        if scaler is None:
            self.scaler.fit(data)
        data = self.scaler.transform(data)
        subjects = all_subjects[indices]
        df = df[df["participant_id"].isin(subjects)]
        self.flatten = flatten
        self.transform = transform
        self.data = np.expand_dims(data, axis=1)
        self.subjects = subjects
        self.n_subjects = len(self.data)
        self.df = df
        print(f"EUAIMS data: {self.data.shape}")
        print(f"EUAIMS data dynamic: {self.data.min()} - {self.data.max()}")
        print(f"EUAIMS subjects: {self.subjects.shape}")
        print(f"EUAIMS metadata: {self.df.shape}")
        print(self.df)

        self.controls_indices = (df["asd"].values == 1)
        self.controls = self.data[self.controls_indices]
        self.patients_indices = (df["asd"].values == 2)
        self.patients = self.data[self.patients_indices]
        print(f"EUAIMS controls: {self.controls.shape}")
        print(f"EUAIMS patients: {self.patients.shape}")
        if self.flatten:
            self.n_samples = len(self.controls) + len(self.patients)
        else:
            self.n_samples = min(len(self.controls), len(self.patients))
            self.reset_mapping()

    def reset_mapping(self):
        """ Reset the controls <-> patients mapping.
        """
        self.controls_mapping = np.random.choice(
            len(self.controls), self.n_samples, replace=False)
        self.patients_mapping = np.random.choice(
            len(self.patients), self.n_samples, replace=False)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.flatten:
            image1 = self.data[idx]
            image2 = np.zeros((0, ))
        else:
            pidx = self.patients_mapping[idx]
            cidx = self.controls_mapping[idx]
            image1 = self.patients[pidx]
            image2 = self.controls[cidx]
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1.astype(np.single), image2.astype(np.single)
