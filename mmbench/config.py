# -*- coding: utf-8 -*
##########################################################################
# NSAp - Copyright (C) CEA, 2023
# Olivier Cornelis
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Configuration parser.
"""

# Imports
import os
import copy
from types import SimpleNamespace


class ConfigParser(object):
    """ Load the specified benchmark configuration.
    """
    def __init__(self, name, configfile):
        """ Init class.

        Parameters
        ----------
        name: str
            the name of the config to be loaded.
        configfile: str
            the path to the config file to be loaded.
        """
        self.name = name
        self.configfile = configfile
        config = {}
        with open(self.configfile) as open_file:
            exec(open_file.read(), config)
        self.config = SimpleNamespace(models=config["_models"])

    def set_auto_params(self, params, default_params):
        """ Set automatically the 'auto' parameters.

        Parameters
        ----------
        params: dict
            the input parameters.
        default_parameters: dict
            the default parameters.

        Returns
        -------
        params: dict
            the filled parameters.
        """
        params = copy.deepcopy(params)
        for name, val in params.items():
            if val != "auto":
                continue
            if name not in default_params:
                raise ValueError(
                    f"Impossible to set default parameter '{name}'")
            params[name] = default_params[name]
        return params
