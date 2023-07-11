# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define some utility functions.
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ModalityExtractor(TransformerMixin, BaseEstimator):
    """ Select the requested modalitites.
    """
    def __init__(self, indices, modalities=None):
        """ Init class.

        Parameters
        ----------
        indices: dict
            the indices to select specific modalities.
        modalities: list of str, default None
            the desired modalities.
            self.modalities = modalities
        """
        self.indices = indices
        self.modalities = modalities or list(indices.keys())

    def fit(self, X, y=None):
        """ Mock method. Does nothing.
        """
        return self

    def transform(self, X, y):
        """ Return a slice of the input array.
        """
        X = get_modalities(X, self.indices, self.modalities)
        return X.copy()

    def fit_transform(self, X, y):
        """ Return a slice of the input array.
        """
        return self.transform(X=X, y=y)


def filter_data(X, y, col_name, min_threshold=None, max_threshold=None):
    """ Remove subjects with score below or upper user-defined thresholds.

    Parameters
    ----------
    X: array (n_samples, n_features)
        the input data.
    y: DataFrame (n_sample, n_labels)
        the variables.
    col_name: str
        the name of the column containing the information to threshold.
    min_threshold: int, default None
        the minimum value. If None no thresholding is applied.
    max_threshold: int, default None
        the maximum value. If None no thresholding is applied.

    Returns
    -------
    X: array (n_selected_samples, n_features)
        the selected data.
    y: DataFrame (n_selected_samples, n_labels)
        the associated variables.
    """
    assert col_name in y.columns, "Can't find the given variable."
    if min_threshold is not None:
        indices = (y[col_name].values > min_threshold).tolist()
        y = y.loc[indices]
        X = X[indices]
    if max_threshold is not None:
        indices = (y[col_name].values < max_threshold).tolist()
        y = y.loc[indices]
        X = X[indices]
    return X.copy(), y.copy()


def get_modalities(X, indices, modalities):
    """ Select the requested modalitites.

    Parameters
    ----------
    X: array or DataFrame (n_subjects, n_features)
        the input ROI data.
    indices: dict
        the indices to select specific modalities.
    modalities: list of str
        the desired modalities.

    Returns
    -------
    X_select: array (n_subjects, n_selected_features)
        the selected ROI data.
    """
    select_indices = []
    for name in modalities:
        select_indices.extend(indices[name])
    if isinstance(X, np.ndarray):
        return X[:, select_indices]
    elif isinstance(X, pd.DataFrame):
        return X.loc[:, X.columns.to_numpy()[select_indices]]
    else:
        raise NotImplementedError


def is_categorial(array_like):
    """ Check if input array contains categorial data.
    """
    return len(np.unique(array_like)) < 50


def sanitize_subjects(array_like):
    """ Sanitize the subject identifiers.
    """
    return [
        int(name.replace("sub-", "").split("_")[0])
        if isinstance(name, str) and name.startswith("sub-") else name
        for name in array_like]


def digitize(values, method="auto"):
    """ Digitize input data.
    """
    bins = np.histogram_bin_edges(values, bins=method)
    new_values = np.digitize(values, bins=bins[1:], right=True)
    return new_values


def listify(obj):
    """ Listify input data.
    """
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]
