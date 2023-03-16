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
    """
    nz1 = self._GiNOD._num_opn_P1
    nz2 = self._GiNOD._num_opn_P2

    # Creates the optimization problem.
    opti = Opti()

    # Gets the players' most likely opinions.
    opn_ego = np.argmax(softmax(z_ego))
    opn_opp = np.argmax(softmax(z_opp))

    # Gets the opponent's control.
    if self._player_id == 1:
      U_ego = subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      U_opp = subgames[opn_ego][opn_opp]._best_operating_point[1][1]
    elif self._player_id == 2:
      U_opp = subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      U_ego = subgames[opn_ego][opn_opp]._best_operating_point[1][1]
    U_ego = np.asarray(U_ego)
    U_opp = np.asarray(U_opp)
    u_ego_nom = U_ego[:, :1]
    u_opp = U_opp[:, :1]

    # Declares the decision variable (control).
    nu = len(self._W_ctrl)
    u_ego = opti.variable(nu,)  # ego's control.

    # Computes the next joint state.
    if self._player_id == 1:
      u_jnt = vertcat(u_ego, u_opp)
    elif self._player_id == 2:
      u_jnt = vertcat(u_opp, u_ego)
    x_next = self._ph_sys.disc_time_dyn_cas(x[np.newaxis].T, u_jnt)

    # Sets the objective function.
    if self._player_id == 1:
      z1 = z_ego
      z2 = z_opp
    elif self._player_id == 2:
      z1 = z_opp
      z2 = z_ego

    J = (u_ego - u_ego_nom).T @ self._W_ctrl @ (u_ego-u_ego_nom)
    # J = u_ego.T @ self._W_ctrl @ u_ego
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

        J += softmax(z1, l1) * softmax(z2, l2) * ((x_next - xnom).T @ Z_ego
                                                  @ (x_next-xnom)
                                                  + zeta_ego.T @ (x_next-xnom))

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
    Assumes two player.
    """
    nz1 = self._GiNOD._num_opn_P1
    nz2 = self._GiNOD._num_opn_P2

    # Creates a grid of ego's actions.
    nu = len(self._W_ctrl)
    config = self._config
    A_MIN = config.A_MIN
    A_MAX = config.A_MAX
    W_MIN = config.W_MIN
    W_MAX = config.W_MAX
    _res_a = 0.015
    _res_w = 0.002

    # N_grid = 10000
    # key = jax.random.PRNGKey(758493)
    # us_ego = jax.random.uniform(
    #     key, shape=((nu, 10000)), minval=jnp.array([[A_MIN, W_MIN]]).T,
    #     maxval=jnp.array([[A_MAX, W_MAX]]).T
    # )

    us_ego = jnp.mgrid[A_MIN:A_MAX:_res_a, W_MIN:W_MAX:_res_w]
    N_grid = us_ego.shape[1] * us_ego.shape[2]
    us_ego = us_ego.reshape(nu, N_grid)

    # Gets the players' most likely opinions.
    opn_ego = np.argmax(softmax(z_ego))
    opn_opp = np.argmax(softmax(z_opp))

    # Gets the opponent's control.
    if self._player_id == 1:
      U_ego = subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      U_opp = subgames[opn_ego][opn_opp]._best_operating_point[1][1]
    elif self._player_id == 2:
      U_opp = subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      U_ego = subgames[opn_ego][opn_opp]._best_operating_point[1][1]
    U_ego = np.asarray(U_ego)
    U_opp = np.asarray(U_opp)
    u_ego_nom = U_ego[:, 0]
    u_opp = U_opp[:, :1]

    us_opp = np.tile(u_opp, N_grid)
    if self._player_id == 1:
      us = np.vstack((us_ego, us_opp))
    elif self._player_id == 2:
      us = np.vstack((us_opp, us_ego))

    # Computes the next physical state.
    xs_next = self._disc_dyn_vmap(x, us)

    # Computes the next opinion state.
    if self._player_id == 1:
      zs_next = self._GiNOD_dyn_vmap(
          xs_next, z_ego, z_opp, att_ego, att_opp, None, subgame_k
      )
    elif self._player_id == 2:
      zs_next = self._GiNOD_dyn_vmap(
          xs_next, z_opp, z_ego, att_opp, att_ego, None, subgame_k
      )

    # Evaluates the cost and picks the best action.
    Z1_k, Z2_k, zeta1_k, zeta2_k, xnom_k, _, _ = subgame_k
    xnom11 = xnom_k[:, 0, 0]
    xnom12 = xnom_k[:, 0, 1]
    xnom21 = xnom_k[:, 1, 0]
    xnom22 = xnom_k[:, 1, 1]
    if self._player_id == 1:
      Z_k = Z1_k
      zeta_k = zeta1_k
    elif self._player_id == 2:
      Z_k = Z2_k
      zeta_k = zeta2_k
    Z11 = Z_k[:, :, 0, 0]
    Z12 = Z_k[:, :, 0, 1]
    Z21 = Z_k[:, :, 1, 0]
    Z22 = Z_k[:, :, 1, 1]
    zeta11 = zeta_k[:, 0, 0]
    zeta12 = zeta_k[:, 0, 1]
    zeta21 = zeta_k[:, 1, 0]
    zeta22 = zeta_k[:, 1, 1]
    _cost_params = (
        u_ego_nom, xnom11, xnom12, xnom21, xnom22, Z11, Z12, Z21, Z22, zeta11,
        zeta12, zeta21, zeta22
    )

    costs = self._l1_cost_vmap(
        xs_next, zs_next[:nz1, :], zs_next[nz1:nz1 + nz2, :], us_ego,
        _cost_params
    )

    idx_min = jnp.argmin(costs)

    # Gets the optimal action.
    u_ego = us_ego[:, idx_min]
    u_ego = np.asarray(u_ego)

    # print(u_ego, u_ego_nom)

    return u_ego
