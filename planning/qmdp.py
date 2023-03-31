"""
RHC planning for Opinion Games.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from casadi import *

import jax
from functools import partial
from jax import jit
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp

from .utils import softmax


class QMDP(object):

  def __init__(self, ph_sys, GiNOD, W_ctrl, player_id, config):
    """
    Initializer.
    """
    self._ph_sys = ph_sys
    self._GiNOD = GiNOD
    self._W_ctrl = W_ctrl  # list of control weighting matrices
    self._player_id = player_id
    self._look_ahead = config.LOOK_AHEAD
    self._config = config

    # vmap functions for level-1-QMDP
    self._disc_dyn_vmap = jit(
        jax.vmap(
            self._ph_sys.disc_time_dyn_jitted, in_axes=(None, 1), out_axes=(1)
        )
    )

    self._GiNOD_dyn_vmap = jit(
        jax.vmap(
            self._GiNOD.disc_dyn_jitted,
            in_axes=(1, None, None, None, None, None, None), out_axes=(1)
        )
    )

    self._l1_cost_vmap = jit(
        jax.vmap(
            self.level_1_cost_jitted, in_axes=(1, 1, 1, 1, None), out_axes=(0)
        )
    )

  @partial(jit, static_argnums=(0,))
  def level_1_cost_jitted(self, x, z1, z2, u_ego, _cost_params):

    def softmax(z: DeviceArray, idx: int) -> DeviceArray:
      return jnp.exp(z[idx]) / jnp.sum(jnp.exp(z))

    def value_func(z1, z2, x, xnom, Z, zeta, idx1, idx2):
      return softmax(z1, idx1) * softmax(
          z2, idx2
      ) * (x - xnom).T @ Z @ (x-xnom) + zeta.T @ (x-xnom)

    (
        unom, xnom11, xnom12, xnom21, xnom22, Z11, Z12, Z21, Z22, zeta11,
        zeta12, zeta21, zeta22
    ) = _cost_params

    J = (u_ego - unom).T @ self._W_ctrl @ (u_ego-unom) + value_func(
        z1, z2, x, xnom11, Z11, zeta11, 0, 0
    ) + value_func(z1, z2, x, xnom12, Z12, zeta12, 0, 1) + value_func(
        z1, z2, x, xnom21, Z21, zeta21, 1, 0
    ) + value_func(z1, z2, x, xnom22, Z22, zeta22, 1, 1)

    return J

  def plan_level_0(
      self, x: np.ndarray, z_ego: np.ndarray, z_opp: np.ndarray, subgames: list
  ) -> np.ndarray:
    """
    Level-0 QMDP planning.
    Assumes two player.

    Args:
        x (np.ndarray): current state
        z_ego (np.ndarray): ego's opinion
        z_opp (np.ndarray): opponent's opinion
        subgames (list): subgames

    Returns:
        np.ndarray: ego's control
    """
    nz1 = self._GiNOD._num_opn_P1
    nz2 = self._GiNOD._num_opn_P2

    # Creates the optimization problem.
    opti = Opti()

    # Gets the ego's most likely opinions.
    opn_ego = np.argmax(softmax(z_ego))
    opn_opp = np.argmax(softmax(z_opp))

    # Gets the opponent's subgame controls and ego's nominal control.
    if self._player_id == 1:
      u_opp11 = np.asarray(subgames[0][0]._best_operating_point[1][1])[:, :1]
      u_opp12 = np.asarray(subgames[0][1]._best_operating_point[1][1])[:, :1]
      u_opp21 = np.asarray(subgames[1][0]._best_operating_point[1][1])[:, :1]
      u_opp22 = np.asarray(subgames[1][1]._best_operating_point[1][1])[:, :1]

      u_ego_nom = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      )[:, :1]

    elif self._player_id == 2:
      u_opp11 = np.asarray(subgames[0][0]._best_operating_point[1][0])[:, :1]
      u_opp12 = np.asarray(subgames[0][1]._best_operating_point[1][0])[:, :1]
      u_opp21 = np.asarray(subgames[1][0]._best_operating_point[1][0])[:, :1]
      u_opp22 = np.asarray(subgames[1][1]._best_operating_point[1][0])[:, :1]

      u_ego_nom = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][1]
      )[:, :1]

    # Declares the decision variable (ego's control).
    nu = len(self._W_ctrl)
    u_ego = opti.variable(nu,)

    # Computes the subgame joint states.
    if self._player_id == 1:
      u_jnt11 = vertcat(u_ego, u_opp11)
      u_jnt12 = vertcat(u_ego, u_opp12)
      u_jnt21 = vertcat(u_ego, u_opp21)
      u_jnt22 = vertcat(u_ego, u_opp22)

    elif self._player_id == 2:
      u_jnt11 = vertcat(u_opp11, u_ego)
      u_jnt12 = vertcat(u_opp12, u_ego)
      u_jnt21 = vertcat(u_opp21, u_ego)
      u_jnt22 = vertcat(u_opp22, u_ego)

    x_next11 = self._ph_sys.disc_time_dyn_cas(x[np.newaxis].T, u_jnt11)
    x_next12 = self._ph_sys.disc_time_dyn_cas(x[np.newaxis].T, u_jnt12)
    x_next21 = self._ph_sys.disc_time_dyn_cas(x[np.newaxis].T, u_jnt21)
    x_next22 = self._ph_sys.disc_time_dyn_cas(x[np.newaxis].T, u_jnt22)

    x_next = [[x_next11, x_next12], [x_next21, x_next22]]

    # Sets the objective function.
    if self._player_id == 1:
      z1 = z_ego
      z2 = z_opp
    elif self._player_id == 2:
      z1 = z_opp
      z2 = z_ego

    J = (u_ego - u_ego_nom).T @ self._W_ctrl @ (u_ego-u_ego_nom)
    # J = 0.
    for l1 in range(nz1):
      for l2 in range(nz2):
        solver = subgames[l1][l2]
        xs_ILQ = np.asarray(solver._best_operating_point[0])
        xnom = xs_ILQ[:, self._look_ahead]
        xnom = xnom[np.newaxis].T
        Zs = np.asarray(solver._best_operating_point[4])[:, :, :, 0]
        zetas = np.asarray(solver._best_operating_point[5])[:, :, 0]
        if self._player_id == 1:
          Z_ego = Zs[0, :, :]
          zeta_ego = zetas[0, :]
        elif self._player_id == 2:
          Z_ego = Zs[1, :, :]
          zeta_ego = zetas[1, :]
        zeta_ego = zeta_ego[np.newaxis].T

        J += softmax(z1, l1) * softmax(z2, l2) * (
            (x_next[l1][l2] - xnom).T @ Z_ego @ (x_next[l1][l2] - xnom)
            + zeta_ego.T @ (x_next[l1][l2] - xnom)
        )

    opti.minimize(J)

    # Defines control constraints. TODO: generalize to arbitrary u dimension.
    config = self._config
    opti.subject_to(opti.bounded(config.A_MIN, u_ego[0, :], config.A_MAX))
    opti.subject_to(opti.bounded(config.W_MIN, u_ego[1, :], config.W_MAX))

    # Solves the optimization.
    opts = {
        "expand": True,
        "ipopt.max_iter": 500,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.bound_frac": 0.5,
        "ipopt.acceptable_iter": 5,
        # "ipopt.linear_solver": "ma57"
    }

    # Disables solver reports.
    opts.update({'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

    # Sets numerical backend.
    opti.solver("ipopt", opts)

    try:
      sol = opti.solve()  # actual solve
    except:
      sol = opti.debug

    u_ego_sol = sol.value(u_ego)
    u_ego_sol = u_ego_sol.reshape(-1)

    return u_ego_sol

  def plan_level_1(
      self, x: np.ndarray, z_ego: np.ndarray, z_opp: np.ndarray,
      att_ego: np.ndarray, att_opp: np.ndarray, subgames: list,
      subgame_k: tuple
  ) -> np.ndarray:
    """
    Level-1 QMDP planning.
    Assumes two player two option.

    Args:
        x (np.ndarray): current state
        z_ego (np.ndarray): ego's opinion
        z_opp (np.ndarray): opponent's opinion
        att_ego (np.ndarray): ego's attention
        att_opp (np.ndarray): opponent's attention
        subgames (list): subgames
        subgame_k (tuple): optimized subgame parameters

    Returns:
        np.ndarray: ego's control
    """

    def softmax_cas(z, idx):
      return exp(z[idx]) / sum1(exp(z))

    nz1 = self._GiNOD._num_opn_P1
    nz2 = self._GiNOD._num_opn_P2

    # Creates the optimization problem.
    opti = Opti()

    # Gets the ego's most likely opinions.
    opn_ego = np.argmax(softmax(z_ego))
    opn_opp = np.argmax(softmax(z_opp))

    # Gets the opponent's subgame controls and ego's nominal control.
    if self._player_id == 1:
      uo_0 = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][1]
      )[:, :1]

      uo11_1 = np.asarray(subgames[0][0]._best_operating_point[1][1])[:, 1:2]
      uo12_1 = np.asarray(subgames[0][1]._best_operating_point[1][1])[:, 1:2]
      uo21_1 = np.asarray(subgames[1][0]._best_operating_point[1][1])[:, 1:2]
      uo22_1 = np.asarray(subgames[1][1]._best_operating_point[1][1])[:, 1:2]

      ue_nom_0 = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      )[:, :1]
      ue_nom_1 = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      )[:, 1:2]

    elif self._player_id == 2:
      uo_0 = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      )[:, :1]

      uo11_1 = np.asarray(subgames[0][0]._best_operating_point[1][0])[:, 1:2]
      uo12_1 = np.asarray(subgames[0][1]._best_operating_point[1][0])[:, 1:2]
      uo21_1 = np.asarray(subgames[1][0]._best_operating_point[1][0])[:, 1:2]
      uo22_1 = np.asarray(subgames[1][1]._best_operating_point[1][0])[:, 1:2]

      ue_nom_0 = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][1]
      )[:, :1]
      ue_nom_1 = np.asarray(
          subgames[opn_ego][opn_opp]._best_operating_point[1][1]
      )[:, 1:2]

    # Declares the decision variable (ego's control).
    nu = len(self._W_ctrl)
    u_ego = opti.variable(nu, 2)

    # Computes the subgame joint states.
    if self._player_id == 1:
      u_jnt_0 = vertcat(u_ego[:, :1], uo_0)

      u_jnt11_1 = vertcat(u_ego[:, 1:], uo11_1)
      u_jnt12_1 = vertcat(u_ego[:, 1:], uo12_1)
      u_jnt21_1 = vertcat(u_ego[:, 1:], uo21_1)
      u_jnt22_1 = vertcat(u_ego[:, 1:], uo22_1)

    elif self._player_id == 2:
      u_jnt_0 = vertcat(uo_0, u_ego[:, :1])

      u_jnt11_1 = vertcat(uo11_1, u_ego[:, 1:])
      u_jnt12_1 = vertcat(uo12_1, u_ego[:, 1:])
      u_jnt21_1 = vertcat(uo21_1, u_ego[:, 1:])
      u_jnt22_1 = vertcat(uo22_1, u_ego[:, 1:])

    x_next_1 = self._ph_sys.disc_time_dyn_cas(x[np.newaxis].T, u_jnt_0)

    x_next11_2 = self._ph_sys.disc_time_dyn_cas(x_next_1, u_jnt11_1)
    x_next12_2 = self._ph_sys.disc_time_dyn_cas(x_next_1, u_jnt12_1)
    x_next21_2 = self._ph_sys.disc_time_dyn_cas(x_next_1, u_jnt21_1)
    x_next22_2 = self._ph_sys.disc_time_dyn_cas(x_next_1, u_jnt22_1)

    x_next_2 = [[x_next11_2, x_next12_2], [x_next21_2, x_next22_2]]

    # Computes the next opinions.
    if self._player_id == 1:
      z1 = z_ego
      z2 = z_opp
      att1 = att_ego
      att2 = att_opp

    elif self._player_id == 2:
      z1 = z_opp
      z2 = z_ego
      att1 = att_opp
      att2 = att_ego

    z_next = self._GiNOD.disc_dyn_two_player_casadi(
        x_next_1, z1, z2, att1, att2, None, subgame_k
    )
    z1_next = z_next[:2]
    z2_next = z_next[2:]

    # Sets the objective function.
    J = (u_ego[:, 0] - ue_nom_0).T @ self._W_ctrl @ (u_ego[:, 0] - ue_nom_0)
    J += (u_ego[:, 1] - ue_nom_1).T @ self._W_ctrl @ (u_ego[:, 1] - ue_nom_1)

    for l1 in range(nz1):
      for l2 in range(nz2):
        solver = subgames[l1][l2]
        xs_ILQ = np.asarray(solver._best_operating_point[0])
        xnom = xs_ILQ[:, self._look_ahead + 1]
        xnom = xnom[np.newaxis].T
        Zs = np.asarray(solver._best_operating_point[4])[:, :, :, 1]
        zetas = np.asarray(solver._best_operating_point[5])[:, :, 1]
        if self._player_id == 1:
          Z_ego = Zs[0, :, :]
          zeta_ego = zetas[0, :]
        elif self._player_id == 2:
          Z_ego = Zs[1, :, :]
          zeta_ego = zetas[1, :]

        J += softmax_cas(z1_next, l1) * softmax_cas(z2_next, l2) * (
            (x_next_2[l1][l2] - xnom).T @ Z_ego @ (x_next_2[l1][l2] - xnom)
            + zeta_ego[np.newaxis] @ (x_next_2[l1][l2] - xnom)
        )

    opti.minimize(J)

    # Defines control constraints. TODO: generalize to arbitrary u dimension.
    config = self._config
    opti.subject_to(opti.bounded(config.A_MIN, u_ego[0, :], config.A_MAX))
    opti.subject_to(opti.bounded(config.W_MIN, u_ego[1, :], config.W_MAX))

    # Solves the optimization.
    opts = {
        "expand": True,
        "ipopt.max_iter": 500,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.bound_frac": 0.5,
        "ipopt.acceptable_iter": 5,
        # "ipopt.linear_solver": "ma57"
    }

    # Disables solver reports.
    opts.update({'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

    # Sets numerical backend.
    opti.solver("ipopt", opts)

    try:
      sol = opti.solve()  # actual solve
    except:
      sol = opti.debug

    u_ego_sol = sol.value(u_ego[:, 0])
    u_ego_sol = u_ego_sol.reshape(-1)

    return u_ego_sol
