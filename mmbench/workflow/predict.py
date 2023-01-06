# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define the predicction workflows.
"""


def benchmark_pred_exp(dataset, datasetdir, outdir):
    """ Compare the learned latent space of different models using
    prediction analysis.

    Parameters
    ----------
    dataset: str
        the dataset name: euaims or hbn.
    datasetdir: str
        the path to the dataset associated data.
    outdir: str
        the destination folder.

    Note
    ----
    The samples are generated with the 'bench-latent' sub-command and are
    stored in the 'outdir' in a file named 'latent_vecs.npz'. The samples
    shape is (n_samples, n_subjects, latent_dim). All samples must have the
    same number of samples and subjects, but possibly different latent
    dimensions.
    """
    raise NotImplementedError
