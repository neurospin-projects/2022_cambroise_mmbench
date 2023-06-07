# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define timeserie clustering functions.
"""

# Imports
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def ts_clustering(X, max_clusters=10, area=None):
    """ Time series clustering.

    Parameters
    ----------
    X: array (N, t)
        the input time series.
    max_clusters: int, default 10
        the maximum number of clusters.
    area: array (N, ), default None
        if set, use the area and start-end of each time serie.

    Returns
    -------
    labels: array (N, )
        the generated labels.
    n_cluster: int
        the number of clusters.
    scores: array (max_clusters, )
        the model selection scores.
    """
    from kneed import KneeLocator
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMinMax

    scores = []
    n_clusters = range(1, max_clusters + 1)
    for n_cluster in n_clusters:
        if area is None:
            X_train = TimeSeriesScalerMinMax().fit_transform(X)
            model = TimeSeriesKMeans(
                n_clusters=n_cluster, metric="dtw", max_iter=20,
                random_state=42)
            model.fit(X_train)
            y_pred = model.labels_
            scores.append(model.inertia_)
        else:
            X_train = StandardScaler().fit_transform(X)
            model = GaussianMixture(
                n_components=n_cluster, covariance_type="full")
            model.fit(X_train)
            y_pred = model.predict(X_train)
            scores.append(model.bic(X_train))
    if area is None:
        n_cluster = KneeLocator(n_clusters, scores, curve="convex",
                                direction="decreasing").knee
    else:
        n_cluster = n_clusters[np.argmin(scores)]
    return y_pred, n_cluster, scores
