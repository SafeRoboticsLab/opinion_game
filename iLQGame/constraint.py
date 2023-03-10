"""
Constraints.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)

TODO:
  - Rewrite comments
"""

import jax.numpy as jnp


class Constraint(object):
  """
  Base class for all constraints. Supports clipping an np.array to satisfy the
  constraint.
  """

  def __init__(self):
    pass

  def clip(self, u):
    """
    Clip the input `u` to satisfy the constraint.
    NOTE: `u` should be a column vector.

    :param u: control input
    :type u: np.array
    :return: clipped input
    :rtype: np.array
    """
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
    NOTE: `u` should be a column vector.

    :param u: control input
    :type u: np.array
    :return: clipped input
    :rtype: np.array
    """
    return jnp.clip(u, self._lower, self._upper)
