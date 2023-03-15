"""
RHC planning for Opinion Games.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np

from .qmdp import QMDP


class RHCPlanner(object):

  def __init__(
      self, subgames, N_sim, ph_sys, ph_sys_casadi, GiNOD, method='QMDPL0',
      config=None, W_ctrl=None
  ):
    """
    Initializer.
    """
    self._subgames = subgames
    self._N_sim = N_sim
    self._ph_sys = ph_sys
    self._GiNOD = GiNOD
    self._method = method
    self._QMDP_P1 = QMDP(
        ph_sys_casadi, GiNOD, W_ctrl[0], player_id=1, config=config
    )
    self._QMDP_P2 = QMDP(
        ph_sys_casadi, GiNOD, W_ctrl[1], player_id=2, config=config
    )
    self._look_ahead = config.LOOK_AHEAD

  def plan(self, x0, z0):
    """
    RHC planning.
    Assumes two player.
    """

    # Initialization.
    nx = len(x0)
    nz = len(z0)
    nz1 = self._GiNOD._num_opn_P1
    nz2 = self._GiNOD._num_opn_P2

    xs = np.zeros((nx, self._N_sim + 1))
    zs = np.zeros((nz, self._N_sim + 1))
    xs[:, 0] = x0
    zs[:, 0] = z0

    Hs = np.zeros((nz1 + nz2, nz1 + nz2, self._N_sim))
    PoI = np.zeros((2, self._N_sim))

    for k in range(self._N_sim):

      # Initialize subgame information.
      Z1_k = np.zeros((nx, nx, nz1, nz2))
      Z2_k = np.zeros((nx, nx, nz1, nz2))
      zeta1_k = np.zeros((nx, nz1, nz2))
      zeta2_k = np.zeros((nx, nz1, nz2))
      xnom_k = np.zeros((nx, nz1, nz2))

      # Solve subgames and collects subgame information.
      for l1 in range(nz1):
        for l2 in range(nz2):
          solver = self._subgames[l1][l2]
          solver.run(xs[:, k])  # solves the subgame.
          xs_ILQ = np.asarray(solver._best_operating_point[0])
          xnom_k[:, l1, l2] = xs_ILQ[:, self._look_ahead]
          Zs = np.asarray(solver._best_operating_point[4])[:, :, :, 0]
          zetas = np.asarray(solver._best_operating_point[5])[:, :, 0]
          Z1_k[:, :, l1, l2] = Zs[0, :, :]
          Z2_k[:, :, l1, l2] = Zs[1, :, :]
          zeta1_k[:, l1, l2] = zetas[0, :]
          zeta2_k[:, l1, l2] = zetas[1, :]

          if k == 0:
            print('[RHC] Subgame', l1, l2, 'compiled.')

      if k == 0:
        z1_k = z0[:nz1]
        z2_k = z0[nz1:nz1 + nz2]
      else:
        z1_k = zs[:nz1, k - 1]
        z2_k = zs[nz1:nz1 + nz2, k - 1]

      # Solves QMDP based on current subgames and opinion states.
      if self._method == 'QMDPL0':
        # Player 1
        u1 = self._QMDP_P1.plan_level_0(xs[:, k], z1_k, z2_k, self._subgames)

        # Player 2
        u2 = self._QMDP_P2.plan_level_0(xs[:, k], z2_k, z1_k, self._subgames)

      elif self._method == 'QMDPL1':
        raise NotImplementedError
      elif self._method == 'QMDPL1L0':
        raise NotImplementedError
      else:
        raise NotImplementedError

      u_list = [u1, u2]

      # Evolves GiNOD.
      x_jnt = np.hstack((xs[:, k], zs[:, k]))
      subgame_k = (Z1_k, Z2_k, zeta1_k, zeta2_k, xnom_k, z1_k, z2_k)

      z_dot_k, H_k, PoI1_k, PoI2_k = self._GiNOD.cont_time_dyn(
          x_jnt, None, subgame_k
      )

      zs[:, k + 1] = zs[:, k] + self._GiNOD._T * np.asarray(z_dot_k)
      Hs[:, :, k] = np.asarray(H_k)
      PoI[:, k] = np.array((PoI1_k, PoI2_k))

      # Evolves physical states.
      x_ph_next = self._ph_sys.disc_time_dyn(xs[:, k], u_list)
      xs[:, k + 1] = np.asarray(x_ph_next)

      # print(x_jnt)
      print(np.round(zs[:, k], 2), PoI1_k, PoI2_k)

    self.xs = xs
    self.zs = zs
    self.Hs = Hs
    self.PoI = PoI
