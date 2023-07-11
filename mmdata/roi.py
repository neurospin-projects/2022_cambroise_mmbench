# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define how to acess ROI datasets.
"""

# Imports
import numpy as np
import pandas as pd
from sklearn import preprocessing
from .utils import is_categorial, listify, sanitize_subjects


def get_roi_data(dtypes, demographic_file, demographic_map=None,
                 encode_map=None, subject_col="participant_id"):
    """ Load the requested ROI data.

    Parameters
    ----------
    dtypes: dict
        path to the ROI TSV or CSV file organized by modality.
    demographic_file: str
        path to a TSV or CSV file containing the demographic information.
    demographic_map: dict
        mapping used to select and rename the important demographic
        information.
    encode_map: dict
        mapping used to re-encode demographic information.
    subject_col: str, default 'participant_id'
        the name of the column containing the subject identifiers.

    Returns
    -------
    X: DataFrame (n_subjects, n_features)
        the loaded ROI data.
    y: DataFrame (n_subjects, n_features)
        the loaded demographic information.
    indices: dict
        the indices to select specific modalities.
    classes: dict
        the encoded categories.
    """
    demographic_map = demographic_map or {}
    y = load_table(demographic_file)
    assert subject_col in y.columns
    y.rename(columns=demographic_map, inplace=True)
    y.replace(encode_map or {}, inplace=True)
    if demographic_map is not None:
        y = y[[subject_col] + list(demographic_map.values())]
    y.dropna(inplace=True)
    classes = {}
    for key in y.columns:
        if key == subject_col:
            continue
        if is_categorial(y[key].values):
            le = preprocessing.LabelEncoder()
            y[key] = le.fit_transform(y[key].values)
            classes[key] = le.classes_

    indices = {}
    offset, X = 0, None
    for key, roi_files in dtypes.items():
        assert key in ROI_LOADERS
        dfs = [ROI_LOADERS[key](path) for path in listify(roi_files)]
        if len(dfs) == 1:
            _X = dfs[0]
        elif len(dfs) == 2:
            assert "_lh_" in roi_files[0]
            assert "_rh_" in roi_files[1]
            dfs[0].columns = [f"{name}_{key}_lh" if idx > 0 else name
                              for idx, name in enumerate(dfs[0].columns)]
            dfs[1].columns = [f"{name}_{key}_rh" if idx > 0 else name
                              for idx, name in enumerate(dfs[1].columns)]
            _X = dfs[0].merge(dfs[1], on="participant_id", how="inner")
        else:
            raise ValueError("Unexpected number of tables.")
        _X.rename(columns={"participant_id": subject_col}, inplace=True)
        n_cols = len(_X.columns) - 1
        indices[key] = list(np.arange(n_cols) + offset)
        offset += n_cols
        _X[subject_col] = sanitize_subjects(_X[subject_col].values)
        if X is None:
            X = _X
        else:
            X = X.merge(_X, on=subject_col, how="inner")
    X.dropna(inplace=True)

    X.set_index(subject_col, inplace=True)
    y[subject_col] = sanitize_subjects(y[subject_col].values)
    y.set_index(subject_col, inplace=True)
    common_subjects = list(set(X.index).intersection(set(y.index)))

    return X.loc[common_subjects], y.loc[common_subjects], indices, classes


def load_table(path):
    """ Load a TSV or CSV table.
    """
    if path.endswith(".tsv"):
        sep = "\t"
    elif path.endswith(".csv"):
        sep = ","
    else:
        raise ValueError("Unexpected extension.")
    return pd.read_csv(path, sep=sep)


def load_cat12_table(path):
    """ Load CAT12 ROI data.
    """
    df = load_table(path)
    keep_colums = [name for name in df.columns if name.endswith("_GM_Vol")]
    keep_colums.insert(0, "participant_id")
    df = df[keep_colums]
    return df


def load_freesurfer_table(path):
    """ Load FreeSurfer ROI data.
    """
    df = load_table(path)
    keep_colums = [name for name in df.columns if "_" in name or "." in name]
    df = df[keep_colums]
    df.rename(columns={keep_colums[0]: "participant_id"}, inplace=True)
    return df


ROI_LOADERS = {
    "vbm": load_cat12_table,
    "thick": load_freesurfer_table,
    "curv": load_freesurfer_table,
    "area": load_freesurfer_table}
