"""
Cost container.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil)
"""

from functools import partial
from jax import jit
from jaxlib.xla_extension import DeviceArray


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

        :param cost: cost function to add
        :type cost: Cost
        :param arg: argument of cost, either "x" or a player index
        :type arg: string or uint
        :param weight: multiplicative weight for this cost
        :type weight: float
        """
    self._costs.append(cost)
    self._args.append(arg)
    self._weights.append(weight)

  # ---------------------------- Jitted functions ------------------------------
  @partial(jit, static_argnums=(0,))
  def get_cost(
      self, x: DeviceArray, ui: DeviceArray, k: int = 0
  ) -> DeviceArray:
    """
    Evaluates this cost function on the given input state.

    Args:
        x (DeviceArray): concatenated state of all subsystems (nx,)
        ui (DeviceArray): control of the subsystem (nui,)
        k (int, optional): time step. Defaults to 0.

    Returns:
        DeviceArray: cost (scalar)
    """
    total_cost = 0.
    for cost, weight in zip(self._costs, self._weights):
      total_cost += weight * cost.get_traj_cost(x, ui)
    return total_cost

  @partial(jit, static_argnums=(0,))
  def quadraticize_jitted(
      self, x: DeviceArray, ui: DeviceArray
  ) -> DeviceArray:
    """
    Calculates the gradients along x and ui.

    Args:
        x (DeviceArray): concatenated state of all subsystems (nx, N)
        ui (DeviceArray): control of the subsystem (nui, N)

    Returns:
        DeviceArray: cost (N,)
        DeviceArray: gradient dc/dx (nx, N)
        DeviceArray: gradient dc/dui (nui, N)
        DeviceArray: Hessian w.r.t. x (nx, nx, N)
        DeviceArray: Hessian w.r.t. ui (nui, nui, N)
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
