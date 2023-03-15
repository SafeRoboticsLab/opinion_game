"""
Util functions for planning.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from casadi import vertcat


class TwoCar8D(object):
  """
  Two car model for casadi.

  TODO: Move to dynamica_system.py
  """

  def __init__(self, l=3.0, T=0.1):
    self._l = l  # inter-axle length (m)
    self._T = T

  def cont_time_dyn_cas(self, x, u):
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x: state vector (8-by-1 opti/MX variable)
        u: control vector (4-by-1 opti/MX variable)
    """
    x0_dot = x[3, 0] * np.cos(x[2, 0])
    x1_dot = x[3, 0] * np.sin(x[2, 0])
    x2_dot = x[3, 0] * np.tan(u[1, 0]) / self._l
    x3_dot = u[0, 0]
    x4_dot = x[7, 0] * np.cos(x[6, 0])
    x5_dot = x[7, 0] * np.sin(x[6, 0])
    x6_dot = x[7, 0] * np.tan(u[3, 0]) / self._l
    x7_dot = u[2, 0]
    return vertcat(
        x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot
    )

  def disc_time_dyn_cas(self, x, u):
    """
    Computes the next state in discrete time.

    Args:
        x: state vector (8-by-1 opti/MX variable)
        u: control vector (4-by-1 opti/MX variable)
    """
    return x + self._T * self.cont_time_dyn_cas(x, u)


def softmax(z: np.ndarray, idx: int = None) -> float:
  """
  Softmax operator.

  Args:
      z (np.ndarray): vector.
      idx (int): index.

  Returns:
      float: output.
  """
  if idx is None:
    return np.exp(z) / np.sum(np.exp(z))
  else:
    return np.exp(z[idx]) / np.sum(np.exp(z))
