# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
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


def residualize(train_df, train_data, test_df, test_data, formula_res=None,
                formula_full=None, site_name=None, discrete_vars=None,
                continuous_vars=None):
    """ Custom linear and/or site Combat residualization.

    Parameters
    ----------
    formula_res: str
        what we want to residualize for 'age + sex + site' or 'age + sex' using
        linear regression.
    formula_full: str
        what we want to adjusted for e.g 'age + sex + diagnosis' or
        'age + sex + site + diagnosis' using linear regression.
    train_df: DataFrame
        table defining the terms used in `formula_full`.
    train_data: array (n_samples, n_features)
        the training data.
    test_df: DataFrame
        table defining the terms used in `formula_full`.
    test_data: array (n_samples, n_features)
        the test data to be transformed.
    site_name: str, default None
        the name of the column containing the site information.
    discrete_vars: list of str, default None
        the name of the covariates which are categorical.
    continuous_vars: list of str, default None
        the name of the covariates which are continuous.

    Returns
    -------
    train_data: array (n_samples, n_features)
        the residualize training data.
    test_data: array (n_samples, n_features)
        the residualize test data.
    """
    from mulm.residualizer import Residualizer
    from neurocombat_sklearn import CombatModel

    all_df = [train_df, test_df]
    sizes = [len(train_df), len(test_df)]
    train_indices = np.arange(sizes[0])
    test_indices = np.arange(sizes[1]) + sizes[0]
    all_df = pd.concat(all_df, ignore_index=True)

    if formula_full is not None:
        residualizer = Residualizer(data=all_df, formula_res=formula_res,
                                    formula_full=formula_full)
        design_matrix = residualizer.get_design_mat(all_df)
        train_data = residualizer.fit_transform(
            train_data, design_matrix[train_indices])
        test_data = residualizer.transform(
            test_data, design_matrix[test_indices])
    if discrete_vars is not None or continuous_vars is not None:
        residualizer = CombatModel()
        residualizer.fit(
            train_data, all_df[site_name].values[train_indices].reshape(-1, 1),
            discrete_covariates=all_df[discrete_vars].values[train_indices],
            continuous_covariates=all_df[continuous_vars].values[
                train_indices])
        train_data = residualizer.transform(
            train_data, all_df[site_name].values[train_indices].reshape(-1, 1),
            discrete_covariates=all_df[discrete_vars].values[train_indices],
            continuous_covariates=all_df[continuous_vars].values[
                train_indices])
        missing_sites = (set(all_df[site_name].values[test_indices]) -
                         set(all_df[site_name].values[train_indices]))
        if len(missing_sites) > 0:
            print(f"Sites '{missing_sites}' were not seen during ComBat "
                  "fit(). Applying transform() for this test set is "
                  "impossible.")
        else:
            test_data = residualizer.transform(
                test_data, all_df[site_name].values[
                    test_indices].reshape(-1, 1),
                discrete_covariates=all_df[discrete_vars].values[test_indices],
                continuous_covariates=all_df[continuous_vars].values[
                    test_indices])

    return train_data, test_data
