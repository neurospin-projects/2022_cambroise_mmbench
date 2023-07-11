# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define base estimator compatible with DataFrame.
"""

# Imports
import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression


class LogisticRegressionDF(LogisticRegression):
    """ Logistic Regression (aka logit, MaxEnt) classifier.
    """
    def __init__(
            self,
            y_col,
            penalty="l2",
            *,
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver="lbfgs",
            max_iter=100,
            multi_class="auto",
            verbose=0,
            warm_start=False,
            n_jobs=None,
            l1_ratio=None):
        self.y_col = y_col
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                         fit_intercept=fit_intercept,
                         intercept_scaling=intercept_scaling,
                         class_weight=class_weight, random_state=random_state,
                         solver=solver, max_iter=max_iter,
                         multi_class=multi_class, verbose=verbose,
                         warm_start=warm_start, n_jobs=n_jobs,
                         l1_ratio=l1_ratio)

    def fit(self, X, y):
        """ Fit the model.

        Parameters
        ----------
        X: array (n_samples, n_features)
            the input data.
        y: DataFrame (n_sample, n_labels)
            the variables.

        Returns
        -------
        self: object
            fitted scaler.
        """
        return super().fit(X, y[self.y_col].values)


class EstimatorDF(TransformerMixin, BaseEstimator):
    """ Use DataFrame as inputs.
    Need to define state cd BaseEstimator.
    """
    def __init__(self, estimator, y_col):
        """ Initialize class.

        Parameters
        ----------
        estimator: BaseEstimator
            the estimator instance.
        y_col: str,
            the name of the column containing the prediction data.
        """
        self.estimator = estimator
        self.y_col = y_col

    def __getattribute__(self, name):
        if name in ("estimator", "y_col", "fit", "__sklearn_clone__"):
            return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self.estimator, name)

    def __setattr__(self, name, value):
        if name in ("estimator", "y_col"):
            return object.__setattr__(self, name, value)
        else:
            return object.__setattr__(self.estimator, name, value)

    def __sklearn_clone__(self):
        estimator = self.estimator.__sklearn_clone__()
        return EstimatorDF(estimator, self.y_col)

    def fit(self, X, y):
        """ Fit the model.

        Parameters
        ----------
        X: array (n_samples, n_features)
            the data used to compute the residualizations.
        y: DataFrame (n_sample, n_labels)
            the variables.

        Returns
        -------
        self: object
            fitted scaler.
        """
        return self.estimator.fit(X, y[self.y_col].values)
