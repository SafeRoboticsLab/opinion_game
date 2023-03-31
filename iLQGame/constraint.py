"""
Constraints.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)
"""

import jax.numpy as jnp


class Constraint(object):
  """
  Base class for all constraints.
  """

  def __init__(self):
    pass

  def clip(self, u):
    raise NotImplementedError("clip is not implemented.")


class BoxConstraint(Constraint):
  """
  Box constraint, derived from Constraint base class.
  """

  def __init__(self, lower, upper):
    self._lower = lower
    self._upper = upper

  def clip(self, u):
    """
    Clip the input `u` to satisfy the constraint.

    Args:
        u (np.ndarray): control input

    Returns:
        np.ndarray: clipped input
    """
    return jnp.clip(u, self._lower, self._upper)
