"""
Jaxified iterative LQ solver.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil)

TODO:
  - Remove logger and visualizer
  - Rewrite comments
"""

import time
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from functools import partial
from jax import jit
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp

from .cost import *


class ILQSolver(object):

  def __init__(
      self, dynamics, player_costs, x0, Ps, alphas, alpha_scaling=0.05,
      max_iter=100, logger=None, visualizer=None, u_constraints=None,
      verbose=False
  ):
    """
        Initialize from dynamics, player costs, current state, and initial
        guesses for control strategies for both players.
        """
    self._dynamics = dynamics
    self._player_costs = player_costs
    self._max_iter = max_iter
    self._x0 = x0
    self._Ps = Ps
    self._alphas = alphas
    self._u_constraints = u_constraints
    self._horizon = Ps[0].shape[2]
    self._num_players = dynamics._num_players

    # Current and previous operating points (states/controls) for use
    # in checking convergence.
    self._last_operating_point = None
    _current_x = jnp.zeros((self._dynamics._x_dim, self._horizon))
    _current_u = [
        jnp.zeros((ui_dim, self._horizon)) for ui_dim in self._dynamics._u_dims
    ]
    self._current_operating_point = (_current_x, _current_u, Ps, alphas)
    self._best_operating_point = (_current_x, _current_u, Ps, alphas)
    self._best_social_cost = 1e10

    # Fixed step size for the linesearch.
    self._alpha_scaling = alpha_scaling

    # Set up visualizer.
    self._visualizer = visualizer
    self._logger = logger

    # Log some of the paramters.
    if self._logger is not None:
      self._logger.log("alpha_scaling", self._alpha_scaling)
      self._logger.log("horizon", self._horizon)
      self._logger.log("x0", self._x0)

    self._verbose = verbose

  def run(self):
    """ Run the algorithm for the specified parameters. """

    iteration = 0

    # while not self._is_converged():
    while iteration <= self._max_iter:

      t_start = time.time()

      # (1) Compute current operating point and update last one.
      tt = time.time()
      current_xs = self._current_operating_point[0]
      current_us = self._current_operating_point[1]
      current_Ps = self._Ps

      # Line search based on social cost.
      total_cost_best = 1e10
      best_alphas = self._alphas
      for alpha_scaling in self._alpha_scaling:
        current_alphas = self._alphas
        current_alphas = [
            (alpha_vec * alpha_scaling) for alpha_vec in current_alphas
        ]
        xs_alpha, us_list_alpha, costs_alpha = self._compute_operating_point(
            current_xs, current_us, current_Ps, current_alphas
        )
        total_cost_alpha = sum([jnp.sum(costis) for costis in costs_alpha])

        if total_cost_alpha < total_cost_best:
          xs = xs_alpha
          us_list = us_list_alpha
          best_alphas = current_alphas
          total_cost_best = total_cost_alpha

      if self._verbose or iteration == 0:
        print("- Reference computing time: ", time.time() - tt)

      # (2) Linearizes about this operating point.
      tt = time.time()
      As, Bs_list = self._linearize_dynamics(xs, us_list)
      if self._verbose or iteration == 0:
        print("- Linearization computing time: ", time.time() - tt)

      # (3) Quadraticize costs.
      tt = time.time()
      costs, lxs, Hxxs, Huus = self._quadraticize_costs(xs, us_list)
      if self._verbose or iteration == 0:
        print("- Quadraticization time: ", time.time() - tt)

      # (4) Compute feedback Nash equilibrium of the resulting LQ game.
      tt = time.time()
      As = np.asarray(As)
      Bs_list = [np.asarray(Bs) for Bs in Bs_list]
      lxs = [np.asarray(lxs_i) for lxs_i in lxs]
      Hxxs = [np.asarray(Hxxs_i) for Hxxs_i in Hxxs]
      Huus = [np.asarray(Huus_i) for Huus_i in Huus]
      Ps, alphas, Zs, zetas = self._solve_lq_game(As, Bs_list, Hxxs, lxs, Huus)
      if self._verbose or iteration == 0:
        print("- Forward & backward pass time: ", time.time() - tt)

      # Accumulates total costs for both players.
      total_costs = [jnp.sum(costis) for costis in costs]
      print(
          "Iteration", iteration, "| Total cost for all players: ",
          total_costs, " | Iter. time: ",
          time.time() - t_start, "\n"
      )

      # Update policy parameters.
      self._Ps = Ps
      self._alphas = alphas

      # Updates the operating points.
      self._last_operating_point = self._current_operating_point
      self._current_operating_point = (
          xs, us_list, current_Ps, best_alphas, Zs, zetas
      )

      if total_cost_best < self._best_social_cost:
        self._best_operating_point = self._current_operating_point
        self._best_social_cost = total_cost_best

      iteration += 1

    plt.close()

  # ------- TODO: CHANGE TO COST DIFFERENCE -------
  def _is_converged(self):
    """ Check if the last two operating points are close enough. """
    if self._last_operating_point is None:
      return False

    # Tolerance for comparing operating points. If all states changes
    # within this tolerance in the Euclidean norm then we've converged.
    TOLERANCE = 1e-4
    for ii in range(self._horizon):
      last_x = self._last_operating_point[0][ii]
      current_x = self._current_operating_point[0][ii]

      if np.linalg.norm(last_x - current_x) > TOLERANCE:
        return False

    return True

  @partial(jit, static_argnums=(0,))
  def _compute_operating_point(
      self, current_xs: DeviceArray, current_us: list, Ps: list, alphas: list
  ) -> Tuple[DeviceArray, list]:
    """
    Computes current operating point by propagating through dynamics.

    Args:
        current_xs (DeviceArray): states (nx, N)
        current_us (list of DeviceArray): controls [(nui, N)]
        Ps (list of DeviceArray)
        alphas (list of DeviceArray)

    Returns:
        DeviceArray: states (nx, N)
        [DeviceArray]: list of controls for each player [(nui, N)]
    """
    x_dim = self._dynamics._x_dim
    u_dims = self._dynamics._u_dims

    # Computes the joint state trajectory.
    xs = jnp.zeros((x_dim, self._horizon))
    xs = xs.at[:, 0].set(self._x0)
    us_list = [jnp.zeros((ui_dim, self._horizon)) for ui_dim in u_dims]

    # NOTE: Here we have to index on the list for disc_time_dyn().
    # So we cannot use fori_loop.
    for k in range(self._horizon):

      x_ref = current_xs[:, k]

      # Computes the control policy for each player.
      us_k = []
      for ii in range(self._num_players):
        # Computes the feedback strategy.
        uii_ref = current_us[ii][:, k]
        uii = uii_ref - Ps[ii][:, :, k] @ (xs[:, k] - x_ref) - alphas[ii][:, k]

        # Project to the control bound.
        uii = self._u_constraints[ii].clip(uii)

        us_k.append(uii)
        us_list[ii] = us_list[ii].at[:, k].set(uii)

      # Computes the next state for the joint system.
      xs = xs.at[:, k + 1].set(self._dynamics.disc_time_dyn(xs[:, k], us_k))

    xs = xs[:, :self._horizon]

    # Evaluates costs.
    costs_list = [[] for _ in range(self._num_players)]
    for ii in range(self._num_players):
      costs_list[ii] = self._player_costs[ii].get_cost(xs, us_list[ii])

    return xs, us_list, costs_list

  @partial(jit, static_argnums=(0,))
  def _linearize_dynamics(self, xs: DeviceArray,
                          us_list: list) -> Tuple[DeviceArray, list]:
    """
    Linearizes dynamics at the current operating point.

    Args:
        xs (DeviceArray): states (nx, N)
        us_list (list of DeviceArray): controls [(nui, N)]

    Returns:
        DeviceArray: A matrices (nx, nx, N)
        [DeviceArray]: list of B matrices for each player [(nx, nui, N)]
    """
    x_dim = self._dynamics._x_dim
    u_dims = self._dynamics._u_dims
    As = jnp.zeros((x_dim, x_dim, self._horizon))
    Bs_list = [jnp.zeros((x_dim, ui_dim, self._horizon)) for ui_dim in u_dims]

    # NOTE: Here we have to index on the list for linearize_discrete_jitted().
    # So we cannot use fori_loop.
    for k in range(self._horizon):
      A, B = self._dynamics.linearize_discrete_jitted(
          xs[:, k], [us_list[ii][:, k] for ii in range(self._num_players)]
      )
      As = As.at[:, :, k].set(A)

      for ii in range(self._num_players):
        Bs_list[ii] = Bs_list[ii].at[:, :, k].set(B[ii])

    return As, Bs_list

  @partial(jit, static_argnums=(0,))
  def _quadraticize_costs(
      self, xs: DeviceArray, us_list: list
  ) -> Tuple[list, list, list, list, list]:
    """
    Quadraticizes costs at the current operating point.

    Args:
        xs (DeviceArray): states (nx, N)
        us_list (list of DeviceArray): controls [(nui, N)]

    Returns:
        [DeviceArray (dtype=float)]: list of costs for each player
        [DeviceArray]: list of gradients lx = dc/dx for each player
        [DeviceArray]: list of Hessians Hxx for each player
        [DeviceArray]: list of Hessians Huu for each player
    """
    costs = [[] for _ in range(self._num_players)]
    lxs = [[] for _ in range(self._num_players)]
    Hxxs = [[] for _ in range(self._num_players)]
    Huus = [[] for _ in range(self._num_players)]
    for ii in range(self._num_players):
      costs_ii, lxs_ii, _, Hxxs_ii, Huus_ii = self._player_costs[
          ii].quadraticize_jitted(xs, us_list[ii])

      costs[ii] = costs_ii
      lxs[ii] = lxs_ii
      Hxxs[ii] = Hxxs_ii
      Huus[ii] = Huus_ii

    return costs, lxs, Hxxs, Huus

  def _solve_lq_game(
      self, As: np.ndarray, Bs_list: list, Qs_list: list, ls_list: list,
      Rs_list: list
  ) -> Tuple[list, list, list, list]:
    """
    Solves a time-varying, finite horizon LQ game (finds closed-loop Nash
    feedback strategies for both players).
    Assumes that dynamics are given by
            ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```

    Args:
        As (np.ndarray): A matrices (nx, nx, N)
        Bs_list ([np.ndarray]): list of players' B matrices [(nui, nui, N)]
        Qs_list ([np.ndarray]): list of Hessians Hxx for each player
        ls_list ([np.ndarray]): list of gradients lx = dc/dx for each player
        Rs_list ([np.ndarray]): list of Hessians Huu for each player

    Returns:
        [np.ndarray]: Ps_list
        [np.ndarray]: alphas_list
        [np.ndarray]: Zs
        [np.ndarray]: zetas
    """

    # Unpack horizon and number of players.
    horizon = self._horizon
    num_players = self._num_players

    # Cache dimensions of state and controls for each player.
    x_dim = self._dynamics._x_dim
    u_dims = self._dynamics._u_dims

    # Note: notation and variable naming closely follows that introduced in
    # the "Preliminary Notation for Corollary 6.1" section, which may be found
    # on pp. 279 of Basar and Olsder.
    # NOTE: we will assume that `c` from Basar and Olsder is always `0`.

    # Recursively computes all intermediate and final variables.
    Zs = [np.zeros((x_dim, x_dim, horizon + 1)) for _ in range(num_players)]
    zetas = [np.zeros((x_dim, horizon + 1)) for _ in range(num_players)]
    for ii in range(num_players):
      Zs[ii][:, :, -1] = Qs_list[ii][:, :, -1]
      zetas[ii][:, -1] = ls_list[ii][:, -1]
    Ps = [np.zeros((uii_dim, x_dim, horizon)) for uii_dim in u_dims]
    alphas = [np.zeros((uii_dim, horizon)) for uii_dim in u_dims]

    # Unpacks lifting matrices.
    LMu = [np.asarray(LMu_ii) for LMu_ii in self._dynamics._LMu]

    for k in range(horizon - 1, -1, -1):
      # Unpacks all relevant variables.
      A = As[:, :, k]
      B = [Bis[:, :, k] for Bis in Bs_list]
      Q = [Qis[:, :, k] for Qis in Qs_list]
      l = [lis[:, k] for lis in ls_list]
      R = [Ris[:, :, k] for Ris in Rs_list]
      Z = [Zis[:, :, k + 1] for Zis in Zs]
      zeta = [zetais[:, k + 1] for zetais in zetas]

      # Computes Ps given previously computed Zs.
      # Refer to equation 6.17a in Basar and Olsder.
      # This will involve solving a system of matrix linear equations of the
      # form [S1s; S2s; ...] * [P1; P2; ...] = [Y1; Y2; ...].
      S_rows = [[] for _ in range(num_players)]
      for ii in range(num_players):
        Sis = [[] for _ in range(num_players)]
        for jj in range(num_players):
          # Sis[jj] = B[ii].T @ np.nan_to_num(Z[ii]) @ B[jj]
          Sis[jj] = B[ii].T @ Z[ii] @ B[jj]
        Sis[ii] += R[ii]
        S_rows[ii] = np.concatenate(Sis, axis=1)
      S = np.concatenate(S_rows, axis=0)

      Y = np.concatenate([B[ii].T @ Z[ii] @ A for ii in range(num_players)])

      P, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
      # P = np.linalg.pinv(S) @ Y

      P_split = [[] for _ in range(num_players)]
      for ii in range(num_players):
        P_split[ii] = LMu[ii] @ P
        Ps[ii][:, :, k] = P_split[ii]

      # Computes F_k = A_k - B1_k P1_k - B2_k P2_k -...
      # This is eq. 6.17c from Basar and Olsder.
      F = A - sum([B[ii] @ P_split[ii] for ii in range(num_players)])

      # Updates Zs.
      for ii in range(num_players):
        Zs[ii][:, :, k
              ] = F.T @ Z[ii] @ F + Q[ii] + P_split[ii].T @ R[ii] @ P_split[ii]

      # Computes alphas using previously computed zetas.
      # Refer to equation 6.17d in Basar and Olsder.
      # This will involve solving a system of linear matrix equations of the
      # form [S1s; S2s; ...] * [alpha1; alpha2; ..] = [Y1; Y2; ...].
      # In fact, this is the same S matrix as before (just a different Y).
      Y = np.concatenate([B[ii].T @ zeta[ii] for ii in range(num_players)])

      alpha, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
      alpha_split = [[] for _ in range(num_players)]
      for ii in range(num_players):
        alpha_split[ii] = LMu[ii] @ alpha
        alphas[ii][:, k] = alpha_split[ii]

      # Computes beta_k = -B1_k alpha1 - B2_k alpha2_k -...
      # This is eq. 6.17f in Basar and Olsder (with `c = 0`).
      beta = -sum([B[ii] @ alpha_split[ii] for ii in range(num_players)])

      # Updates zetas to be the next step earlier in time (now they
      # correspond to time k+1). This is Remark 6.3 in Basar and Olsder.
      for ii in range(num_players):
        zetas[ii][:, k] = F.T @ (zeta[ii] + Z[ii] @ beta) + l[
            ii] + P_split[ii].T @ R[ii] @ alpha_split[ii]
    return Ps, alphas, Zs, zetas

  def plot_controls(self, player_id, fig_width=15., fig_length=8.):
    """
    Plots control signals.

    Args:
        player_id (int): id of the player whose control signal is to be plotted
    """
    dt = self._dynamics._T
    current_us = self._current_operating_point[1]

    u1 = current_us[player_id - 1][0, :].flatten()
    u2 = current_us[player_id - 1][1, :].flatten()

    t_grid = np.arange(0, current_us[player_id - 1].shape[1]) * dt

    plt.figure()
    fig, axs = plt.subplots(2)
    fig.suptitle(
        'Optimized controls of Player ' + str(player_id) + ' over time'
    )
    fig.set_size_inches(fig_width, fig_length)

    axs[0].step(t_grid, u1)
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('$\omega$ (rad/s)')
    axs[0].grid()

    axs[1].step(t_grid, u2)
    axs[1].set_xlabel('time (s)')
    axs[1].set_ylabel('$a$ (m/s$^2$)')
    axs[1].grid()

    plt.show()

  def plot_opinions(
      self, opn_state_idx_list, fig_width=12., fig_length=10., font_size=16
  ):
    """
    Plots opinion state trajectories.

    Args:
        opn_state_idx (list): [[z1_index, u1_index], [z2_index, u2_index], ...]
    """
    # assert len(opn_state_idx_list) == self._num_players

    dt = self._dynamics._T
    current_xs = self._current_operating_point[0]

    plt.rcParams.update({'font.size': font_size})
    plt.figure()
    fig, axs = plt.subplots(len(opn_state_idx_list), 1)
    fig.set_size_inches(fig_width, fig_length)
    fig.tight_layout(pad=5.0)

    t_grid = np.arange(0, current_xs.shape[1]) * dt

    player_count = 0
    for opn_state_idx in opn_state_idx_list:

      z_idx = opn_state_idx[0]
      u_idx = opn_state_idx[1]

      z = current_xs[z_idx, :].flatten()
      u = current_xs[u_idx, :].flatten()

      if len(opn_state_idx_list) > 1:
        ax = axs[player_count]
      else:
        ax = axs

      ax.plot(t_grid, z, color='b')
      ax.plot(t_grid, u, color='m')
      ax.set_xlabel('time (s)')
      ax.set_ylabel('opinion state')
      ax.title.set_text(
          'Opinion states of Player ' + str(player_count + 1) + ' over time'
      )
      ax.legend(('$z$', '$\lambda$'), loc='upper right')
      ax.grid()

      player_count += 1

    plt.show()