"""
RHC planning for Opinion Games.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np


class RHCPlanner(object):

  def __init__(self, ILQSolver, N_sim, GiNOD, subgame):
    """
    Initializer.
    """
    self._ILQSolver = ILQSolver
    self._N_sim = N_sim
    self._GiNOD = GiNOD

    (Z1, Z2, zeta1, zeta2, xnom) = subgame
    self._Z1 = np.asarray(Z1)
    self._Z2 = np.asarray(Z2)
    self._zeta1 = np.asarray(zeta1)
    self._zeta2 = np.asarray(zeta2)
    self._xnom = np.asarray(xnom)

  def plan(self, x0):
    """
    RHC planning.
    """

    # Initialization.
    xs = x0
    x = x0

    for k in range(self._N_sim):
      x_ph = np.hstack(
          (x[self._GiNOD._x_indices_P1], x[self._GiNOD._x_indices_P2])
      )[:, np.newaxis]

      Z1_k = np.zeros_like(self._Z1[:, :, :, :, 0])
      Z2_k = np.zeros_like(self._Z2[:, :, :, :, 0])
      zeta1_k = np.zeros_like(self._zeta1[:, :, :, 0])
      zeta2_k = np.zeros_like(self._zeta2[:, :, :, 0])
      xnom_k = np.zeros_like(self._xnom[:, :, :, 0])
      # Searches for the nearest state in each subgame trajecotries.
      for l1 in [1, 2]:
        for l2 in [1, 2]:
          xerr = np.linalg.norm(
              x_ph - self._xnom[:, l1 - 1, l2 - 1, :], axis=0
          )
          idx_k = np.argmin(xerr)
          Z1_k[:, :, l1 - 1, l2 - 1] = self._Z1[:, :, l1 - 1, l2 - 1, idx_k]
          Z2_k[:, :, l1 - 1, l2 - 1] = self._Z2[:, :, l1 - 1, l2 - 1, idx_k]
          zeta1_k[:, l1 - 1, l2 - 1] = self._zeta1[:, l1 - 1, l2 - 1, idx_k]
          zeta2_k[:, l1 - 1, l2 - 1] = self._zeta2[:, l1 - 1, l2 - 1, idx_k]
          xnom_k[:, l1 - 1, l2 - 1] = self._xnom[:, l1 - 1, l2 - 1, idx_k]

      # Assigns warmstart strategies.
      Ps_warmstart = None  #self._ILQSolver._Ps
      alphas_warmstart = None  #self._ILQSolver._alphas

      # Solves iLQ-OG.
      z1_k = x[self._GiNOD._z_indices_P1]
      z2_k = x[self._GiNOD._z_indices_P2]
      att1_k = x[self._GiNOD._att_indices_P1]
      att2_k = x[self._GiNOD._att_indices_P2]

      subgame_k = (Z1_k, Z2_k, zeta1_k, zeta2_k, xnom_k, z1_k, z2_k)

      t_start = time.time()
      self._ILQSolver.run_OG_two_player(
          x, subgame_k, Ps_warmstart, alphas_warmstart
      )
      print("[RHC] iLQ-OG solving time: ", time.time() - t_start, "\n")

      x_ILQ = self._ILQSolver._best_operating_point[0][:, 1]
      # print(self._ILQSolver._best_operating_point[0][:, 8:12])

      # **** THERE IS A BUG ****

      # Stores results.
      z = np.hstack((z1_k, z2_k, att1_k, att2_k))
      z_dot_k = self._GiNOD.cont_time_dyn(x, None, 0, subgame_k)
      z = z + self._GiNOD._T * z_dot_k
      x_ph = np.hstack(
          (x_ILQ[self._GiNOD._x_indices_P1], x_ILQ[self._GiNOD._x_indices_P2])
      )
      x = np.hstack((x_ph, z))

      xs = np.vstack((xs, x))

      print(z)

    self.xs = xs
