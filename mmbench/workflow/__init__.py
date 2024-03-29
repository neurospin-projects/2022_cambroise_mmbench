# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Definition of the workflows.
"""

from .smcvae import train_smcvae
from .cvae import train_cvae
from .embedding import benchmark_latent_exp
from .rsa import benchmark_rsa_exp
from .predict import benchmark_pred_exp
from .similarity import benchmark_feature_similarity_exp
from .barrier import benchmark_barrier_exp
from .baseline import benchmark_baseline
