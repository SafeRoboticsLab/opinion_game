"""
Game-induced nonlinear opinion dynamics (GiNOD).

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from typing import Tuple
from casadi import vertcat, horzcat, kron

import jax
from functools import partial
from jax import jit, jacfwd, lax
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp

from .utils import softmax
from iLQGame.dynamical_system import *


class NonlinearOpinionDynamicsTwoPlayer(DynamicalSystem):
  """
  Two Player Nonlinear Opinion Dynamics.

  For jit compatibility, number of players is hardcoded to 2 to avoid loops.
  """

  def __init__(
      self, x_indices_P1, x_indices_P2, z_indices_P1, z_indices_P2,
      att_indices_P1, att_indices_P2, z_P1_bias, z_P2_bias, damping_opn=0.0,
      damping_att=[0.0, 0.0], rho=[1.0, 1.0], T=0.1, z_norm_thresh=10.0
  ):
    """
    Initializer.

    Joint state vector should be organized as
      xi := [x, z_P1, z_P2, lambda_P1, lambda_P2]

    Args:
        x_indices_P1 (DeviceArray, dtype=int32): P1 x (physical states) indices
        x_indices_P2 (DeviceArray, dtype=int32): P2 x (physical states) indices
        z_indices_P1 (DeviceArray, dtype=int32): P1 z (opinion states) indices
        z_indices_P2 (DeviceArray, dtype=int32): P2 z (opinion states) indices
        att_indices_P1 (DeviceArray, dtype=int32): P1 attention indices
        att_indices_P2 (DeviceArray, dtype=int32): P2 attention indices
        z_P1_bias (DeviceArray): (nz,) P1 opinion state bias
        z_P2_bias (DeviceArray): (nz,) P2 opinion state bias
        damping_opn (float, optional): z damping parameter. Defaults to 0.0.
        damping_att (float, optional): att damping parameter. Defaults to 0.0.
        rho (float, optional): att scaling parameter. Defaults to 1.0.
        T (float, optional): time interval. Defaults to 0.1.
    """
    self._x_indices_P1 = x_indices_P1
    self._x_indices_P2 = x_indices_P2
    self._z_indices_P1 = z_indices_P1
    self._z_indices_P2 = z_indices_P2
    self._att_indices_P1 = att_indices_P1
    self._att_indices_P2 = att_indices_P2
    self._z_P1_bias = z_P1_bias
    self._z_P2_bias = z_P2_bias
    self._damping_opn = damping_opn
    self._damping_att = damping_att
    self._rho = rho

    self._eps = 0.
    self._PoI_max = 10.0
    self._z_norm_thresh = z_norm_thresh

    # Players' number of options
    self._num_opn_P1 = len(self._z_indices_P1)
    self._num_opn_P2 = len(self._z_indices_P2)

    self._num_att_P1 = len(self._att_indices_P1)
    self._num_att_P2 = len(self._att_indices_P2)

    self._x_dim = (
        self._num_opn_P1 + self._num_opn_P2 + self._num_att_P1
        + self._num_att_P2
    )

    super(NonlinearOpinionDynamicsTwoPlayer, self).__init__(self._x_dim, 0, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(
      self,
      x: DeviceArray,
      ctrl=None,
      subgame: Tuple = (),
  ) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.
    This is an autonomous system.

    Args:
        x (DeviceArray): (nx,) where nx is the dimension of the joint
          system (physical subsystems plus all players' opinion dynamics)
          For each opinion dynamics, their state := (z, u) where z is the
          opinion state and u is the attention parameter
        ctrl (DeviceArray): None

        subgame (Tuple) include:
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

    Returns:
        DeviceArray: next state (nx,)
        DeviceArray: H (nz, nz)
    """

    def Vhat1(
        z1: DeviceArray, z2: DeviceArray, x_ph: DeviceArray
    ) -> DeviceArray:
      """
      Opinion-weighted game value function for P1.
      """
      V_hat = 0.
      for l1 in range(self._num_opn_P1):
        for l2 in range(self._num_opn_P2):
          xe = x_ph - x_ph_nom[:, l1, l2]  # error state
          Z_sub = Z_P1[:, :, l1, l2]
          zeta_sub = zeta_P1[:, l1, l2]
          nom_cost_sub = nom_cost_P1[l1, l2]
          V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
          V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub
      return V_hat

    def Vhat2(
        z1: DeviceArray, z2: DeviceArray, x_ph: DeviceArray
    ) -> DeviceArray:
      """
      Opinion-weighted game value function for P2.
      """
      V_hat = 0.
      for l1 in range(self._num_opn_P1):
        for l2 in range(self._num_opn_P2):
          xe = x_ph - x_ph_nom[:, l1, l2]  # error state
          Z_sub = Z_P2[:, :, l1, l2]
          zeta_sub = zeta_P2[:, l1, l2]
          nom_cost_sub = nom_cost_P2[l1, l2]
          V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
          V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub
      return V_hat

    def compute_PoI_P1(z1: DeviceArray, x_ph: DeviceArray) -> DeviceArray:
      """
      Computes the Price of Indecision (PoI) for P1.
      """
      ratios = jnp.zeros((self._num_opn_P2,))

      # Outer loop over P2's (opponent) options.
      for l2 in range(self._num_opn_P2):
        V_subs = jnp.zeros((self._num_opn_P1,))

        # Inner loop over P1's (ego) options.
        for l1 in range(self._num_opn_P1):
          xe = x_ph - x_ph_nom[:, l1, l2]  # error state
          Z_sub = Z_P1[:, :, l1, l2]
          zeta_sub = zeta_P1[:, l1, l2]
          nom_cost_sub = nom_cost_P1[l1, l2]
          V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
          V_subs = V_subs.at[l1].set(V_sub)

        # Normalize to avoid large numbers.
        V_subs = jax.nn.softmax(jax.nn.standardize(V_subs))

        numer = 0.
        for l1 in range(self._num_opn_P1):
          numer += softmax(z1, l1) * V_subs[l1]

        denom = jnp.min(V_subs)

        ratio = (numer + self._eps) / (denom + self._eps)
        ratios = ratios.at[l2].set(ratio)

      PoI = jnp.max(ratios)
      return jnp.minimum(PoI, self._PoI_max)

    def compute_PoI_P2(z2: DeviceArray, x_ph: DeviceArray) -> DeviceArray:
      """
      Computes the Price of Indecision (PoI) for P2.
      """
      ratios = jnp.zeros((self._num_opn_P1,))

      # Outer loop over P1's (opponent) options.
      for l1 in range(self._num_opn_P1):
        numer = 0.
        V_subs = jnp.zeros((self._num_opn_P2,))

        # Inner loop over P2's (ego) options.
        for l2 in range(self._num_opn_P2):
          xe = x_ph - x_ph_nom[:, l1, l2]  # error state
          Z_sub = Z_P2[:, :, l1, l2]
          zeta_sub = zeta_P2[:, l1, l2]
          nom_cost_sub = nom_cost_P2[l1, l2]
          V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
          V_subs = V_subs.at[l2].set(V_sub)

        # Normalize to avoid large numbers.
        V_subs = jax.nn.softmax(jax.nn.standardize(V_subs))

        numer = 0.
        for l2 in range(self._num_opn_P2):
          numer += softmax(z2, l2) * V_subs[l2]

        denom = jnp.min(V_subs)

        ratio = (numer + self._eps) / (denom + self._eps)
        ratios = ratios.at[l1].set(ratio)

      PoI = jnp.max(ratios)
      return jnp.minimum(PoI, self._PoI_max)

    def true_fn(PoI):
      return PoI

    def false_fn(PoI):
      return 1.0

    (
        Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2, nom_cost_P1,
        nom_cost_P2
    ) = subgame

    # State variables.
    x_ph1 = x[self._x_indices_P1]
    x_ph2 = x[self._x_indices_P2]
    x_ph = jnp.hstack((x_ph1, x_ph2))

    z1 = x[self._z_indices_P1]
    z2 = x[self._z_indices_P2]
    z = jnp.hstack((z1, z2))

    att1 = x[self._att_indices_P1]
    att2 = x[self._att_indices_P2]

    # Computes game Hessians.
    dVhat1_dz1 = jacfwd(Vhat1, argnums=0)
    H1s = jacfwd(dVhat1_dz1, argnums=[0, 1])
    H1 = jnp.hstack(H1s(znom_P1, znom_P2, x_ph))

    dVhat2_dz2 = jacfwd(Vhat2, argnums=1)
    H2s = jacfwd(dVhat2_dz2, argnums=[0, 1])
    H2 = jnp.hstack(H2s(znom_P1, znom_P2, x_ph))

    H1 = jax.nn.standardize(H1)  # Avoid large numbers.
    H2 = jax.nn.standardize(H2)  # Avoid large numbers.

    # Computes the opinion state time derivative.
    att_1_vec = att1 * jnp.ones((self._num_opn_P1,))
    att_2_vec = att2 * jnp.ones((self._num_opn_P2,))

    D = jnp.diag(
        self._damping_opn * jnp.ones(self._num_opn_P1 + self._num_opn_P2,)
    )

    H1z = att_1_vec * jnp.tanh(H1@z + self._z_P1_bias)
    H2z = att_2_vec * jnp.tanh(H2@z + self._z_P2_bias)

    z_dot = -D @ z + jnp.hstack((H1z, H2z))

    # Computes the attention time derivative.
    PoI_1 = jnp.nan_to_num(compute_PoI_P1(z1, x_ph), nan=1.0)
    PoI_2 = jnp.nan_to_num(compute_PoI_P2(z2, x_ph), nan=1.0)

    z1_norm = jnp.linalg.norm(z1)
    PoI_1 = lax.cond(z1_norm <= self._z_norm_thresh, true_fn, false_fn, PoI_1)

    z2_norm = jnp.linalg.norm(z2)
    PoI_2 = lax.cond(z2_norm <= self._z_norm_thresh, true_fn, false_fn, PoI_2)

    att1_dot = -self._damping_att[0] * att1 + self._rho[0] * (PoI_1-1.0)
    att2_dot = -self._damping_att[1] * att2 + self._rho[1] * (PoI_2-1.0)

    # Joint state time derivative.
    x_jnt_dot = jnp.hstack((z_dot, att1_dot, att2_dot))

    return x_jnt_dot, jnp.vstack((H1, H2)), PoI_1, PoI_2
    # return x_jnt_dot

  @partial(jit, static_argnums=(0,))
  def disc_dyn_jitted(
      self,
      x_ph: DeviceArray,
      z1: DeviceArray,
      z2: DeviceArray,
      att1: DeviceArray,
      att2: DeviceArray,
      ctrl=None,
      subgame: Tuple = (),
  ) -> DeviceArray:
    """
    Computes the next opinion state.
    This is an autonomous system.
    """

    def Vhat1(
        z1: DeviceArray, z2: DeviceArray, x_ph: DeviceArray
    ) -> DeviceArray:
      """
      Opinion-weighted game value function for P1.
      """
      V_hat = 0.
      for l1 in range(self._num_opn_P1):
        for l2 in range(self._num_opn_P2):
          xe = x_ph - x_ph_nom[:, l1, l2]  # error state
          Z_sub = Z_P1[:, :, l1, l2]
          zeta_sub = zeta_P1[:, l1, l2]
          nom_cost_sub = nom_cost_P1[l1, l2]
          V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
          V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub
      return V_hat

    def Vhat2(
        z1: DeviceArray, z2: DeviceArray, x_ph: DeviceArray
    ) -> DeviceArray:
      """
      Opinion-weighted game value function for P2.
      """
      V_hat = 0.
      for l1 in range(self._num_opn_P1):
        for l2 in range(self._num_opn_P2):
          xe = x_ph - x_ph_nom[:, l1, l2]  # error state
          Z_sub = Z_P2[:, :, l1, l2]
          zeta_sub = zeta_P2[:, l1, l2]
          nom_cost_sub = nom_cost_P2[l1, l2]
          V_sub = xe.T @ Z_sub @ xe + zeta_sub.T @ xe + nom_cost_sub
          V_hat += softmax(z1, l1) * softmax(z2, l2) * V_sub
      return V_hat

    (
        Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2, nom_cost_P1,
        nom_cost_P2
    ) = subgame

    # State variables.
    z = jnp.hstack((z1, z2))

    # Computes game Hessians.
    dVhat1_dz1 = jacfwd(Vhat1, argnums=0)
    H1s = jacfwd(dVhat1_dz1, argnums=[0, 1])
    H1 = jnp.hstack(H1s(znom_P1, znom_P2, x_ph))

    dVhat2_dz2 = jacfwd(Vhat2, argnums=1)
    H2s = jacfwd(dVhat2_dz2, argnums=[0, 1])
    H2 = jnp.hstack(H2s(znom_P1, znom_P2, x_ph))

    H1 = jax.nn.standardize(H1)  # Avoid large numbers.
    H2 = jax.nn.standardize(H2)  # Avoid large numbers.

    # Computes the opinion state time derivative.
    att_1_vec = att1 * jnp.ones((self._num_opn_P1,))
    att_2_vec = att2 * jnp.ones((self._num_opn_P2,))

    D = jnp.diag(
        self._damping_opn * jnp.ones(self._num_opn_P1 + self._num_opn_P2,)
    )

    H1z = att_1_vec * jnp.tanh(H1@z + self._z_P1_bias)
    H2z = att_2_vec * jnp.tanh(H2@z + self._z_P2_bias)

    z_dot = -D @ z + jnp.hstack((H1z, H2z))

    return z + self._T * z_dot

  def disc_dyn_two_player_casadi(
      self, x_ph, z1, z2, att1, att2, ctrl=None, subgame: Tuple = ()
  ):
    """
    Computes the next opinion state using the analytical GiNOD.
    For optimization in casadi.
    """

    def softmax(z, idx):
      return np.exp(z[idx]) / np.sum(np.exp(z))

    def phi_a(z):
      return (softmax(z, 0) - softmax(z, 1)) * phi_b(z)

    def phi_b(z):
      return softmax(z, 0) * softmax(z, 1)

    (
        Z_P1, Z_P2, zeta_P1, zeta_P2, x_ph_nom, znom_P1, znom_P2, nom_cost_P1,
        nom_cost_P2
    ) = subgame

    # State variables.
    z = jnp.hstack((z1, z2))

    # Computes subgame value functions.
    zeta1_11 = zeta_P1[:, 0, 0]
    V1_11 = (x_ph - x_ph_nom[:, 0, 0]).T @ Z_P1[:, :, 0, 0] @ (
        x_ph - x_ph_nom[:, 0, 0]
    ) + zeta1_11[np.newaxis] @ (x_ph - x_ph_nom[:, 0, 0]) + nom_cost_P1[0, 0]
    zeta1_12 = zeta_P1[:, 0, 1]
    V1_12 = (x_ph - x_ph_nom[:, 0, 1]).T @ Z_P1[:, :, 0, 1] @ (
        x_ph - x_ph_nom[:, 0, 1]
    ) + zeta1_12[np.newaxis] @ (x_ph - x_ph_nom[:, 0, 1]) + nom_cost_P1[0, 1]
    zeta1_21 = zeta_P1[:, 1, 0]
    V1_21 = (x_ph - x_ph_nom[:, 1, 0]).T @ Z_P1[:, :, 1, 0] @ (
        x_ph - x_ph_nom[:, 1, 0]
    ) + zeta1_21[np.newaxis] @ (x_ph - x_ph_nom[:, 1, 0]) + nom_cost_P1[1, 0]
    zeta1_22 = zeta_P1[:, 1, 1]
    V1_22 = (x_ph - x_ph_nom[:, 1, 1]).T @ Z_P1[:, :, 1, 1] @ (
        x_ph - x_ph_nom[:, 1, 1]
    ) + zeta1_22[np.newaxis] @ (x_ph - x_ph_nom[:, 1, 1]) + nom_cost_P1[1, 1]

    zeta2_11 = zeta_P2[:, 0, 0]
    V2_11 = (x_ph - x_ph_nom[:, 0, 0]).T @ Z_P2[:, :, 0, 0] @ (
        x_ph - x_ph_nom[:, 0, 0]
    ) + zeta2_11[np.newaxis] @ (x_ph - x_ph_nom[:, 0, 0]) + nom_cost_P2[0, 0]
    zeta2_12 = zeta_P2[:, 0, 1]
    V2_12 = (x_ph - x_ph_nom[:, 0, 1]).T @ Z_P2[:, :, 0, 1] @ (
        x_ph - x_ph_nom[:, 0, 1]
    ) + zeta2_12[np.newaxis] @ (x_ph - x_ph_nom[:, 0, 1]) + nom_cost_P2[0, 1]
    zeta2_21 = zeta_P2[:, 1, 0]
    V2_21 = (x_ph - x_ph_nom[:, 1, 0]).T @ Z_P2[:, :, 1, 0] @ (
        x_ph - x_ph_nom[:, 1, 0]
    ) + zeta2_21[np.newaxis] @ (x_ph - x_ph_nom[:, 1, 0]) + nom_cost_P2[1, 0]
    zeta2_22 = zeta_P2[:, 1, 1]
    V2_22 = (x_ph - x_ph_nom[:, 1, 1]).T @ Z_P2[:, :, 1, 1] @ (
        x_ph - x_ph_nom[:, 1, 1]
    ) + zeta2_22[np.newaxis] @ (x_ph - x_ph_nom[:, 1, 1]) + nom_cost_P2[1, 1]

    # Computes game Hessians.
    a1 = phi_a(z1) * (
        softmax(z2, 0) * (V1_11-V1_21) + softmax(z2, 1) * (V1_12-V1_22)
    )
    a2 = phi_a(z2) * (
        softmax(z1, 0) * (V2_11-V2_12) + softmax(z1, 1) * (V2_21-V2_22)
    )
    b1 = phi_b(z1) * phi_b(z2) * (-V1_11 - V1_22 + V1_12 + V1_21)
    b2 = phi_b(z1) * phi_b(z2) * (-V2_11 - V2_22 + V2_12 + V2_21)

    Gamma_1 = horzcat(a1, b1)
    Gamma_2 = horzcat(b2, a2)
    Gamma = vertcat(Gamma_1, Gamma_2)

    Gamma -= (a1+a2+b1+b2) / 4.0  # Normalize Gamma by subtracting the mean.

    H_tmp = np.array([[1, -1], [-1, 1]])
    H = kron(Gamma, H_tmp)

    # Computes the opinion state time derivative.
    att_1_vec = att1 * np.ones((self._num_opn_P1,))
    att_2_vec = att2 * np.ones((self._num_opn_P2,))
    att_vec = np.hstack((att_1_vec, att_2_vec))

    D = np.diag(
        self._damping_opn * np.ones(self._num_opn_P1 + self._num_opn_P2,)
    )

    bias = np.hstack((self._z_P1_bias, self._z_P2_bias))

    z_dot = -D @ z + att_vec * np.tanh(H@z + bias)

    return z + self._T * z_dot
