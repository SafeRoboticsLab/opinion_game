"""
Cost container.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil)
"""

from functools import partial
from jax import jit
from jaxlib.xla_extension import ArrayImpl


class PlayerCost(object):

  def __init__(self):
    self._costs = []
    self._args = []
    self._weights = []

  def add_cost(self, cost, arg, weight=1.0):
    """
    Add a new cost to the game, and specify its argument to be either
    "x" or an integer indicating which player's control it is, e.g. 0
    corresponds to u0. Also assign a weight.

    Args:
        cost (Cost): cost function to add
        arg (string or int): argument of cost, either "x" or a player index
        weight (float, optional): multiplicative weight for this cost
    """
    self._costs.append(cost)
    self._args.append(arg)
    self._weights.append(weight)

  # ---------------------------- Jitted functions ------------------------------
  @partial(jit, static_argnums=(0,))
  def get_cost(self, x: ArrayImpl, ui: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Evaluates this cost function on the given input state.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx,)
        ui (ArrayImpl): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        ArrayImpl: cost (scalar)
    """
    total_cost = 0.
    for cost, weight in zip(self._costs, self._weights):
      total_cost += weight * cost.get_traj_cost(x, ui)
    return total_cost

  @partial(jit, static_argnums=(0,))
  def quadraticize_jitted(self, x: ArrayImpl, ui: ArrayImpl) -> ArrayImpl:
    """
    Calculates the gradients along x and ui.

    Args:
        x (ArrayImpl): concatenated state of all subsystems (nx, N)
        ui (ArrayImpl): control of the subsystem (nui, N)

    Returns:
        ArrayImpl: cost (N,)
        ArrayImpl: gradient dc/dx (nx, N)
        ArrayImpl: gradient dc/dui (nui, N)
        ArrayImpl: Hessian w.r.t. x (nx, nx, N)
        ArrayImpl: Hessian w.r.t. ui (nui, nui, N)
    """
    total_cost = 0.
    total_dcdx = 0.
    total_dcdu = 0.
    total_Hxx = 0.
    total_Huu = 0.
    for cost, weight in zip(self._costs, self._weights):
      # Updates total cost.
      total_cost += weight * cost.get_traj_cost(x, ui)

      # Updates total gradients.
      current_dcdx, current_dcdu = cost.get_traj_grad(x, ui)
      total_dcdx += weight * current_dcdx
      total_dcdu += weight * current_dcdu

      # Updates total Hessians.
      current_Hxx, current_Huu = cost.get_traj_hess(x, ui)
      total_Hxx += weight * current_Hxx
      total_Huu += weight * current_Huu

    return total_cost, total_dcdx, total_dcdu, total_Hxx, total_Huu
