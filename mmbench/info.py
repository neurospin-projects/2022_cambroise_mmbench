# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Benchmark multi-model/multi-view models.
"""
SUMMARY = """
.. container:: summary-carousel

    The availability of multiple data types provides a rich source of
    information and holds promise for learning representations that
    generalize well across multiple modalities. Multimodal data naturally
    grants additional self-supervision in the form of shared information
    connecting the different data types. Further, the understanding of
    different modalities and the interplay between data types are non-trivial
    research questions and long-standing goals in machine learning research.
"""
long_description = (
    "Benchmark multi-model/multi-view models.\n")

# Main setup parameters
NAME = "mmbench"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "NeuroSpin webPage"
EXTRAURL = (
    "https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/"
    "NeuroSpin.aspx")
LINKS = {"projects": "https://github.com/neurospin-projects"}
URL = "https://github.com/rlink7/rlink_7limri"
DOWNLOAD_URL = "https://github.com/neurospin-projects/2022_cambroise_mmbench"
LICENSE = "CeCILL-B"
AUTHOR = """
mmbench developers
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["mmbench"]
REQUIRES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "scikit-learn",
    "fire",
    "torch",
    "torchvision",
    ("brainite @ "
     "git+https://github.com/neurospin-deepinsight/brainite.git#egg=brainite"),
    ("mopoe @ "
     "git+https://github.com/neurospin-deepinsight/mopoe.git#egg=mopoe")
]
SCRIPTS = [
    "mmbench/scripts/mmbench"
]
