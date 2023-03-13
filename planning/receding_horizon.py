"""
RHC planning for Opinion Games.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np
from typing import Tuple

from functools import partial
from jax import jit
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


class RHCPlanner(object):

  def __init__(self, x0, ILQSolver, N_sim):
    """
    Initializer.
    """
    self._x0 = x0
    self._ILQSolver = ILQSolver
    self._N_sim = N_sim

  def plan(self):
    """
    RHC planning
    """
