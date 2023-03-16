"""
Util functions for planning.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from casadi import vertcat

from functools import partial
from jax import jit
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


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

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn_jitted(
      self, x: DeviceArray, u: DeviceArray
  ) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x: state vector (8-by-1 DeviceArray)
        u: control vector (4-by-1 DeviceArray)
    """
    x0_dot = x[3] * jnp.cos(x[2])
    x1_dot = x[3] * jnp.sin(x[2])
    x2_dot = x[3] * jnp.tan(u[1]) / self._l
    x3_dot = u[0]
    x4_dot = x[7] * jnp.cos(x[6])
    x5_dot = x[7] * jnp.sin(x[6])
    x6_dot = x[7] * jnp.tan(u[3]) / self._l
    x7_dot = u[2]
    return jnp.hstack(
        (x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot)
    )

  @partial(jit, static_argnums=(0,))
  def disc_time_dyn_jitted(
      self, x: DeviceArray, u: DeviceArray
  ) -> DeviceArray:
    """
    Computes the one-step evolution of the system in discrete time with Euler
    integration.

    Args:
        x: state vector (8-by-1 DeviceArray)
        u: control vector (4-by-1 DeviceArray)
    """
    x_dot = self.cont_time_dyn_jitted(x, u)
    return x + self._T * x_dot


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
