# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define residualization tools.
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from mulm.residualizer import Residualizer
from neurocombat_sklearn import CombatModel


class MRIScaler(TransformerMixin, BaseEstimator):
    """ Transform features using linear and/or site Combat residualization.
    """
    def __init__(self, formula_res=None, formula_full=None,
                 site_name=None, discrete_vars=None, continuous_vars=None,
                 scale=True):
        """ Initialize class.

        Parameters
        ----------
        formula_res: str, default None
            what we want to residualize for 'age + sex + site' or 'age + sex'
            using linear regression.
        formula_full: str, default None
            what we want to adjusted for 'age + sex + diagnosis' or
            'age + sex + site + diagnosis' using linear regression.
        site_name: str, default None
            the name of the column containing the site information.
        discrete_vars: list of str, default None
            the name of the covariates which are categorical.
        continuous_vars: list of str, default None
            the name of the covariates which are continuous.
        scale: bool, default True
            control the process.
        """
        self.formula_res = formula_res
        self.formula_full = formula_full
        self.site_name = site_name
        self.discrete_vars = discrete_vars
        self.continuous_vars = continuous_vars
        self.residualize = self.formula_full is not None
        self.site_removal = (self.discrete_vars is not None or
                             self.continuous_vars is not None)
        self.scale = scale

    def _reset(self):
        """ Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "residualizer"):
            del self.residualizer
        if hasattr(self, "site_residualizer"):
            del self.site_residualizer
            del self.known_sites
        if hasattr(self, "fitted_"):
            del self.fitted_

    def fit(self, X, y):
        """ Compute the different residualizations.

        Parameters
        ----------
        X: array (n_samples, n_features)
            the data used to compute the residualizations.
        y: DataFrame (n_sample, n_labels)
            the residualization variables.

        Returns
        -------
        self: object
            fitted scaler.
        """
        self._reset()
        if self.scale and self.residualize:
            self.residualizer = Residualizer(
                data=y, formula_res=self.formula_res,
                formula_full=self.formula_full)
            design_matrix = self.residualizer.get_design_mat(y)
            self.residualizer.fit(X, design_matrix)
        if self.scale and self.site_removal:
            self.site_residualizer = CombatModel()
            self.known_sites = set(y[self.site_name].values)
            self.site_residualizer.fit(
                X, y[self.site_name].values.reshape(-1, 1),
                discrete_covariates=y[self.discrete_vars].values,
                continuous_covariates=y[self.continuous_vars].values)
        self.fitted_ = True
        return self

    def transform(self, X, y):
        """ Residualize features of X.

        Parameters
        ----------
        X: array (n_samples, n_features)
            input data that will be transformed.
        y: DataFrame (n_sample, n_labels)
            the residualization variables.

        Returns
        -------
        Xt: array (n_samples, n_features)
            transformed data.
        """
        check_is_fitted(self)
        Xt = X
        if self.scale and self.residualize:
            design_matrix = self.residualizer.get_design_mat(y)
            Xt = self.residualizer.transform(Xt, design_matrix)
        if self.scale and self.site_removal:
            missing_sites = (set(y[self.site_name].values) - self.known_sites)
            if len(missing_sites) > 0:
                print(f"Sites '{missing_sites}' were not seen during ComBat "
                      "fit(). Applying transform() for this test set is "
                      "impossible.")
            else:
                Xt = self.site_residualizer.transform(
                    Xt, y[self.site_name].values.reshape(-1, 1),
                    discrete_covariates=y[self.discrete_vars].values,
                    continuous_covariates=y[self.continuous_vars].values)
        return Xt

    def fit_transform(self, X, y):
        """ Fit parameters and apply residualization.

        Parameters
        ----------
        X: array (n_samples, n_features)
            the data used to compute the residualizations.
        y: DataFrame (n_sample, n_labels)
            the residualization variables.

        Returns
        -------
        Xt: array (n_samples, n_features)
            transformed data.
        """
        self.fit(X, y)
        return self.transform(X, y)
