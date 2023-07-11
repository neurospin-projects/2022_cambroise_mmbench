# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define model selection utility functions.
"""

# Imports
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_cv(X, y, n_splits=5, random_state=43):
    """ Get stratified cross-validation indices.

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
        training data, where n_samples is the number of samples
        and n_features is the number of features.
        Note that providing ``y`` is sufficient to generate the splits and
        hence ``np.zeros(n_samples)`` may be used as a placeholder for
        ``X`` instead of actual training data.
    y: array, shape (n_samples, n_labels)
        the target variable for supervised learning problems.
        Multilabel stratification is done based on the y labels.
    random_state: int, default=43
        if int, random_state is the seed used by the random number generator.

    Returns
    -------
    cv: list
        each list element contains the training and testings sets of indices
        for that split.
    """
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=random_state)
    train_indices, test_indices = [], []
    for i1, i2 in mskf.split(X, y.values):
        train_indices.append(i1)
        test_indices.append(i2)
    cv = zip(train_indices, test_indices)
    return cv
