"""
Building multi-Fidelity emulator using many single-output GP.

1. SingleBinGP: the single-fidelity emulator in the paper.
2. SingleBinLinearGP: the linear multi-fidelity emulator (AR1).
3. SingleBinNonLinearGP: the non-linear multi-fidelity emulator (NARGP).
4. SingleBinDeepGP: the deep GP for multi-fidelity (MF-DGP). This one is not
    mentioned in the paper due to we haven't found a way to fine-tune the
    hyperparameters.

Most of the model constructions are similar to Emukit's examples, with some
modifications on the choice of hyperparameters and modelling each output as
an independent GP (many single-output GP).
"""
from typing import Tuple, List, Optional, Dict

import logging
import numpy as np

from .latin_hypercube import map_to_unit_cube_list

_log = logging.getLogger(__name__)

def _map_params_to_unit_cube(
    params: np.ndarray, param_limits: np.ndarray
) -> np.ndarray:
    """
    Map the parameters onto a unit cube so that all the variations are
    similar in magnitude.
    
    :param params: (n_points, n_dims) parameter vectors
    :param param_limits: (n_dim, 2) param_limits is a list 
        of parameter limits.
    :return: params_cube, (n_points, n_dims) parameter vectors 
        in a unit cube.
    """
    nparams = np.shape(params)[1]
    params_cube = map_to_unit_cube_list(params, param_limits)
    assert params_cube.shape[1] == nparams

    return params_cube