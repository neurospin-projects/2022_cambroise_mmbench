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
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
try:
    from cvae.datasets import ContrastiveDataset
except:
    ContrastiveDataset = object
from mopoe.multimodal_cohort.dataset import MultimodalDataset, DataManager
from mopoe.multimodal_cohort.dataset import MissingModalitySampler
from mmbench.color_utils import print_text
from mmbench.residualize import residualize as residualizer


# Global parameters
IQ_MAP = {
    "euaims": 75.,
    "hbn": None
}


def get_train_data(dataset, datasetdir, modalities, residualize=False):
    """ See `get_data` and `iq_threshold` for documentation.
    """
    threshold = IQ_MAP.get(dataset)
    if dataset == "hbn":
        data, meta_df = get_data_legacy(dataset, datasetdir, modalities,
                                        dtype="train")
    else:
        _, meta_df, data, train_indices, test_indices = get_data(
            dataset, datasetdir, modalities, dtype="complete",
            residualize=residualize)
        meta_df = pd.DataFrame(data=meta_df.values[train_indices],
                               columns=meta_df.columns,
                               index=meta_df.index[train_indices])
        data = dict((key, item.X_train) for key, item in data.items())
    data, meta_df = iq_threshold(dataset, data, meta_df, threshold=threshold)
    return data, meta_df


def get_test_data(dataset, datasetdir, modalities, residualize=False):
    """ See `get_data` and `iq_threshold` for documentation.
    """
    threshold = IQ_MAP.get(dataset)
    if dataset == "hbn":
        data, meta_df = get_data_legacy(dataset, datasetdir, modalities,
                                        dtype="test")
    else:
        _, meta_df, data, train_indices, test_indices = get_data(
            dataset, datasetdir, modalities, dtype="complete",
            residualize=residualize)
        meta_df = pd.DataFrame(data=meta_df.values[test_indices],
                               columns=meta_df.columns,
                               index=meta_df.index[test_indices])
        data = dict((key, item.X_test) for key, item in data.items())
    data, meta_df = iq_threshold(dataset, data, meta_df, threshold=threshold)
    return data, meta_df


def get_train_full_data(dataset, datasetdir, modalities, residualize=False):
    """ See `get_data` and `iq_threshold` for documentation.
    """
    threshold = IQ_MAP.get(dataset)
    if dataset == "hbn":
        data, meta_df = get_data_legacy(dataset, datasetdir, modalities,
                                        dtype="full_train")
    else:
        _, meta_df, data, train_indices, test_indices = get_data(
            dataset, datasetdir, modalities, dtype="full",
            residualize=residualize)
        meta_df = pd.DataFrame(data=meta_df.values[train_indices],
                               columns=meta_df.columns,
                               index=meta_df.index[train_indices])
        data = dict((key, item.X_train) for key, item in data.items())
    data, meta_df = iq_threshold(dataset, data, meta_df, threshold=threshold)
    return data, meta_df


def get_test_full_data(dataset, datasetdir, modalities, residualize=False):
    """ See `get_data` and `iq_threshold` for documentation.
    """
    threshold = IQ_MAP.get(dataset)
    if dataset == "hbn":
        data, meta_df = get_data_legacy(dataset, datasetdir, modalities,
                                        dtype="full_test")
    else:
        _, meta_df, data, train_indices, test_indices = get_data(
            dataset, datasetdir, modalities, dtype="full",
            residualize=residualize)
        meta_df = pd.DataFrame(data=meta_df.values[test_indices],
                               columns=meta_df.columns,
                               index=meta_df.index[test_indices])
        data = dict((key, item.X_test) for key, item in data.items())
    data, meta_df = iq_threshold(dataset, data, meta_df, threshold=threshold)
    return data, meta_df


def iq_threshold(dataset, data, meta_df, threshold=80, col_name="fsiq"):
    """ Remove subjects with IQ below a user-defined threshold.

    Parameters
    ----------
    data: dict
        the loaded data for each modality.
    metadata: DataFrame
        the associated meta information.
    threshold: int, default 80
        the minimum IQ. If None no thresholding is applied.
    col_name: str, default 'fsiq'
        the name of the column containing the IQ information.

    Returns
    -------
    data: dict
        the loaded data thresholded for each modality.
    meta_df: DataFrame
        the associated meta information.
    """
    if threshold is None:
        return data, meta_df
    assert col_name in meta_df.columns, "Can't find the given IQ column name."
    indices = meta_df[col_name].values > threshold
    print_text(f"Filtering data: {np.sum(indices)}/{len(meta_df)}")
    meta_df = meta_df.loc[indices]
    indices = torch.argwhere(torch.from_numpy(indices)).flatten()
    for key, tensor in data.items():
        data[key] = torch.index_select(tensor, 0, indices)
    return data, meta_df


def get_data_legacy(dataset, datasetdir, modalities, dtype):
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
        the data type: 'train', 'test', 'full_test', 'full_train' or 'full'.

    Returns
    -------
    data: dict
        the loaded data for each modality.
    meta_df: DataFrame
        the associated meta information.
    """
    trainset, testset = get_dataset(dataset, datasetdir, modalities)
    if dtype == "train":
        dataset = trainset
    elif dtype == "test":
        dataset = testset
    elif dtype == "full":
        datasets = [trainset, testset]
    elif dtype == "full_test":
        datasets = [testset]
    elif dtype == "full_train":
        datasets = [trainset]
    else:
        raise ValueError("Unexpected data type.")
    if dtype.startswith("full"):
        all_data = {"rois": [], "clinical": []}
        all_meta = None
        for dataset in datasets:
            sampler = MissingModalitySampler(dataset, batch_size=len(dataset))
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
            for data, _, meta in loader:
                if "rois" not in data:
                    continue
                all_data["rois"].append(data["rois"])
                if "clinical" not in data:
                    all_data["clinical"].append(None)
                else:
                    all_data["clinical"].append(data["clinical"])
                if all_meta is None:
                    all_meta = dict((key, [val]) for key, val in meta.items())
                else:
                    for key, val in meta.items():
                        all_meta[key].append(val)
        clinical_size = set([item.size(1) if item is not None else 0
                             for item in all_data["clinical"]])
        if len(clinical_size) > 1:
            clinical_size.remove(0)
        assert len(clinical_size) == 1, "All blocks must have the same size."
        clinical_size = list(clinical_size)[0]
        for idx, (roi_items, clin_items) in enumerate(
                zip(all_data["rois"], all_data["clinical"])):
            if clin_items is None:
                block = torch.empty((roi_items.size(0), clinical_size))
                block[:] = float("nan")
                all_data["clinical"][idx] = block
        all_data["rois"] = torch.cat(all_data["rois"], dim=0)
        all_data["clinical"] = torch.cat(all_data["clinical"], dim=0)
        print(all_data["rois"].shape, all_data["clinical"].shape)
        for key in all_meta:
            all_meta[key] = np.concatenate(all_meta[key], axis=0)
        data, meta = (all_data, all_meta)
    else:
        sampler = MissingModalitySampler(dataset, batch_size=len(dataset))
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        while True:
            dataiter = iter(loader)
            data, _, meta = next(dataiter)
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


def get_data(dataset, datasetdir, modalities, dtype="complete",
             test_size=0.2, residualize=False, random_state=42):
    """ Load the train/test data.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    modalities: list of str
        the modalities to load.
    dtype: str, default 'complete'
        the data type: 'complete', 'full'.
    test_size: float, default=0.2
        should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split.
    residualize: bool, default False
        optionaly residualize the image data.
    random_state: int, default 42
        controls the shuffling applied to the data before applying the split.

    Returns
    -------
    data: dict of DataFrame
        the loaded data for each modality.
    meta_df: DataFrame
        the associated meta information.
    tensors: dict of Tensors
        the splitted input data (train, test).
    train_indices: list of int
        the train indices.
    test_indices: list of int
        the test indices.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    data, meta_df = load_data(datasetdir, modalities)
    data["rois"].dropna(inplace=True)
    if dtype == "full":
        subjects = data["rois"].index
    elif dtype == "complete":
        data["clinical"].dropna(inplace=True)
        subjects = set(data["rois"].index).intersection(
            set(data["clinical"].index))
    else:
        raise ValueError("Unexpected data type.")
    meta_df = meta_df[meta_df.index.isin(subjects)]
    for key, df in data.items():
        data[key] = df[df.index.isin(subjects)]

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(
        msss.split(list(subjects), meta_df.values))
    meta_train_df = pd.DataFrame(data=meta_df.values[train_indices],
                                 columns=meta_df.columns,
                                 index=meta_df.index[train_indices])
    meta_test_df = pd.DataFrame(data=meta_df.values[test_indices],
                                columns=meta_df.columns,
                                index=meta_df.index[test_indices])

    tensors = {}
    for key, df in data.items():
        X_train = df.values[train_indices]
        X_test = df.values[test_indices]
        if residualize and key == "rois":
            X_train, X_test = residualizer(
                meta_train_df, X_train, meta_test_df, X_test,
                formula_res="age + sex",
                formula_full="age + sex + asd", site_name="site",
                discrete_vars=["sex"], continuous_vars=["age"])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        cset = SimpleNamespace(X_train=torch.from_numpy(X_train),
                               X_test=torch.from_numpy(X_test))
        tensors[key] = cset

    return data, meta_df, tensors, train_indices, test_indices


def load_data(datasetdir, modalities):
    """ Load the data.

    Parameters
    ----------
    datasetdir: str
        the path to the dataset associated data.
    modalities: list of str
        the modalities to load.

    Returns
    -------
    data: dict of DataFrame
        the loaded data.
    meta_df: DataFrame
        the associated meta information.
    """
    meta_df = pd.read_csv(os.path.join(datasetdir, "metadata.tsv"), sep="\t")
    meta_df.set_index("participant_id", inplace=True)
    data = {}
    all_subjects = []
    for mod in modalities:
        _data = np.load(os.path.join(datasetdir, f"{mod}_data.npy"))
        subjects = np.load(os.path.join(datasetdir, f"{mod}_subjects.npy"))
        all_subjects.extend(subjects.tolist())
        names = np.load(os.path.join(datasetdir, f"{mod}_names.npy"),
                        allow_pickle=True)
        data[mod] = pd.DataFrame(data=_data, columns=names, index=subjects)
    all_subjects = set(all_subjects)
    for key, df in data.items():
        missing_subjects = all_subjects - set(df.index)
        new_df = pd.DataFrame(np.nan, index=list(missing_subjects),
                              columns=df.columns)
        data[key] = pd.concat([df, new_df])
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


class EUAIMSDataset(ContrastiveDataset):
    """ From the EUAIMS cohort the target and background datasets are composed
    of T1w MRI FreeSurfer ROI features of ASD patients and TD controls,
    respectively.
    """
    def __init__(self, root, train=True, transform=None, flatten=False,
                 seed=42, scaler=None):
        """ Init class.

        Parameters
        ----------
        root: str
            root directory of dataset where data will be saved.
        train: bool, default True
            specifies training or test dataset.
        transform: callable, default None
            optional transform to be applied on a sample.
        flatten: bool, default False
            optionally select all data.
        seed: int, default 42
            for reproducibility fix a seed.
        scaler: sklearn-like scaler, default None
            optionally set a fitted scaler.
        """
        if train and scaler is None:
            scaler = StandardScaler()
        super(EUAIMSDataset, self).__init__(
            root, train, transform, flatten, seed, scaler)

    def get_data(self):
        """ Get the background and target data.

        Returns
        -------
        background: array (N, n_channels, \*)
            the background data.
        background_labels: array (N, )
            the background labels.
        target: array (M, n_channels, \*)
            the target data.
        target_labels: array (M, )
            the target labels.
        """
        split = "train" if self.train else "test"
        meta_split_file = os.path.join(self.root, f"metadata_{split}.tsv")
        roi_file = os.path.join(self.root, "rois_data.npy")
        subject_file = roi_file.replace("_data.npy", "_subjects.npy")
        self.is_file(meta_split_file)
        self.is_file(meta_split_file)
        self.is_file(subject_file)
        df = pd.read_csv(meta_split_file, sep="\t")
        subjects = df["participant_id"].values
        data = np.load(roi_file)
        all_subjects = np.load(subject_file)
        indices = np.nonzero(np.in1d(all_subjects, subjects))[0]
        print(data.shape, len(subjects), len(indices))
        data = data[indices]
        subjects = all_subjects[indices]
        df = df[df["participant_id"].isin(subjects)]
        data = np.expand_dims(data, axis=1)
        print(f"EUAIMS data: {data.shape}")
        print(f"EUAIMS data dynamic: {data.min()} - {data.max()}")
        print(f"EUAIMS subjects: {subjects.shape}")
        print(f"EUAIMS metadata: {df.shape}")
        print(df)
        controls_indices = (df["asd"].values == 1)
        background = data[controls_indices]
        background_labels = np.array(["td"] * len(background))
        patients_indices = (df["asd"].values == 2)
        target = data[patients_indices]
        target_labels = np.array(["asd"] * len(target))
        print(f"EUAIMS controls: {background.shape}")
        print(f"EUAIMS patients: {target.shape}")
        return background, background_labels, target, target_labels

    def is_file(self, path):
        """ Check wethe a EUAIMS data resource file is here.
        """
        if not os.path.isfile(path):
            raise ValueError("The root folder must contains the EUAIMS data.")
