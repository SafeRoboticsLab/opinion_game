"""
Game-induced nonlinear opinion dynamics.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

from typing import Tuple

from functools import partial
from jax import jit, jacfwd
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp

from utils import softmax
from ..iLQGame.dynamical_system import DynamicalSystem


class NonlinearOpinionDynamicsTwoPlayer(DynamicalSystem):
  """
  Two Player Nonlinear Opinion Dynamics.

  For jit compatibility, number of players is hardcoded to 2 to avoid loops.
  """

  def __init__(
      self, dim_P1, dim_P2, x_indices_P1, x_indices_P2, z_indices_P1,
      z_indices_P2, Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2,
      damping=0.0, T=0.1
  ):
    """
    Initializer.

    Joint state vector should be organized as
      xi := [x, z_P1, z_P2, lambda_P1, lambda_P2]

    Args:
        dim_P1 (float): P1 NOD dimension (including attention)
        dim_P2 (float): P2 NOD dimension (including attention)
        x_indices_P1 (DeviceArray, dtype=int32): P1 x (physical states) indices
        x_indices_P2 (DeviceArray, dtype=int32): P2 x (physical states) indices
        z_indices_P1 (DeviceArray, dtype=int32): P1 z (opinion states) indices
        z_indices_P2 (DeviceArray, dtype=int32): P2 z (opinion states) indices
        Z_P1 (DeviceArray): (nx_ph, nx_ph, num_opn_P1, num_opn_P2) P1's Z
          (subgame cost matrices)
        Z_P2 (DeviceArray): (nx_ph, nx_ph, num_opn_P1, num_opn_P2) P2's Z
          (subgame cost matrices)
        zeta_P1 (DeviceArray): (nx_ph, num_opn_P1, num_opn_P2) P1's zeta
          (subgame cost vectors)
        zeta_P2 (DeviceArray): (nx_ph, num_opn_P1, num_opn_P2) P2's zeta
          (subgame cost vectors)
        x_ph_nom (DeviceArray): (nx_ph, num_opn_P1, num_opn_P2) subgame
          nominal physical states
        znom_P1 (DeviceArray): (nz_P1,) P1 nominal z
        znom_P2 (DeviceArray): (nz_P2,) P2 nominal z
        damping (float, optional): damping parameter. Defaults to 0.0.
        T (float, optional): time interval. Defaults to 0.1.
    """
    self._dim_P1 = dim_P1
    self._dim_P1 = dim_P2
    self._x_indices_P1 = x_indices_P1
    self._x_indices_P2 = x_indices_P2
    self._z_indices_P1 = z_indices_P1
    self._z_indices_P2 = z_indices_P2
    self._Z_P1 = Z_P1
    self._Z_P2 = Z_P2
    self._zeta_P1 = zeta_P1
    self._zeta_P2 = zeta_P2
    self._x_ph_nom = x_ph_nom
    self._znom_P1 = znom_P1
    self._znom_P2 = znom_P2
    self._damping = damping

    # Players' number of options
    self._num_opn_P1 = len(self._z_indices_P1)
    self._num_opn_P2 = len(self._z_indices_P2)

    super(NonlinearOpinionDynamicsTwoPlayer,
          self).__init__(dim_P1 + dim_P2, 0, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, ctrl=None) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.
    This is an autonomous system.

    Args:
        x (DeviceArray): (nx,) where nx is the dimension of the joint
          system (physical subsystems plus all players' opinion dynamics)
          For each opinion dynamics, their state := (z, u) where z is the
          opinion state and u is the attention parameter
        ctrl (DeviceArray): None

    Returns:
        DeviceArray: next state (nx,)
    """

    @partial(jit, static_argnums=(3, 4))
    def compute_V_hat(
        z1: DeviceArray, z2: DeviceArray, x_ph: DeviceArray, Z: DeviceArray,
        zeta: DeviceArray
    ) -> DeviceArray:
      """
      Opinion-weighted game value function.
      """
      V_hat = 0.
      for l1 in range(self._num_opn_P1):
        for l2 in range(self._num_opn_P2):
          xe = x_ph - self._x_ph_nom[:, l1, l2]  # error state
          V_sub = xe.T @ Z[:, :, l1, l2] @ xe + zeta[:, :, l1, l2].T @ xe
          V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub
      return V_hat

    # State variables.
    x_ph1 = x[self._x_indices_P1]
    x_ph2 = x[self._x_indices_P2]
    x_ph = jnp.hstack((x_ph1, x_ph2))

    z1 = x[self._z_indices_P1]
    z2 = x[self._z_indices_P2]
    z = jnp.hstack((z1, z2))

    # Computes game Hessians.
    V_hat_P1 = compute_V_hat(z1, z2, x_ph, self._Z_P1, self._zeta_P1)
    V_hat_P2 = compute_V_hat(z1, z2, x_ph, self._Z_P2, self._zeta_P2)

    dV_hat_P1_dz1 = jacfwd(V_hat_P1, argnums=0)

    # TODO attention dynamics.
