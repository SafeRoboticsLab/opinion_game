"""
RHC planning for Opinion Games.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from casadi import *

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
      self, x: np.ndarray, z_ego: np.ndarray, z_opp: np.ndarray, subgames: list
  ) -> np.ndarray:
    """
    Level-1 QMDP planning.
    Assumes two player.
    """
    nz1 = self._GiNOD._num_opn_P1
    nz2 = self._GiNOD._num_opn_P2

    # Creates the optimization problem.
    opti = Opti()

    # Gets the players' most likely opinions.
    opn_ego = np.argmax(softmax(z_ego))
    opn_opp = np.argmax(softmax(z_opp))

    # Gets the opponent's control (first two steps).
    if self._player_id == 1:
      U_ego = subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      U_opp = subgames[opn_ego][opn_opp]._best_operating_point[1][1]
    elif self._player_id == 2:
      U_opp = subgames[opn_ego][opn_opp]._best_operating_point[1][0]
      U_ego = subgames[opn_ego][opn_opp]._best_operating_point[1][1]
    U_ego = np.asarray(U_ego)
    U_opp = np.asarray(U_opp)
    u_ego_nom = U_ego[:, :2]
    u_opp = U_opp[:, :2]

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
