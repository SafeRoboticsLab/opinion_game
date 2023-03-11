"""
NOD test: Two car toll station example.
"""

import os
import time
import numpy as np
import jax.numpy as jnp
from copy import deepcopy

from iLQGame.cost import *
from iLQGame.utils import *
from iLQGame.geometry import *
from iLQGame.constraint import *
from iLQGame.dynamical_system import *
from iLQGame.multiplayer_dynamical_system import *

from iLQGame.ilq_solver import ILQSolver
from iLQGame.player_cost import PlayerCost

from NOD.opinion_dynamics import NonlinearOpinionDynamicsTwoPlayer

# Loads the config.
config = load_config("example_two_car.yaml")

# General parameters.
TIME_HORIZON = config.TIME_HORIZON  # s
TIME_RES = config.TIME_RES  # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RES)
LOG_DIRECTORY = "experiments/two_car"
FILE_NAME = "two_car"
N = int(config.TIME_HORIZON / config.TIME_RES)

# Load subgame results.
Z1 = np.zeros((8, 8, 2, 2, N + 1))
Z2 = np.zeros((8, 8, 2, 2, N + 1))
zeta1 = np.zeros((8, 2, 2, N + 1))
zeta2 = np.zeros((8, 2, 2, N + 1))
xnom = np.zeros((8, 2, 2, N))
for l1 in [1, 2]:
  for l2 in [1, 2]:

    Z_list_sub = np.load(
        os.path.join(
            LOG_DIRECTORY, FILE_NAME + '_' + str(l1) + str(l2) + '_Zs.npy'
        )
    )
    Z1[:, :, l1 - 1, l2 - 1, :] = Z_list_sub[0]
    Z2[:, :, l1 - 1, l2 - 1, :] = Z_list_sub[1]

    zeta_list_sub = np.load(
        os.path.join(
            LOG_DIRECTORY, FILE_NAME + '_' + str(l1) + str(l2) + '_zetas.npy'
        )
    )
    zeta1[:, l1 - 1, l2 - 1, :] = zeta_list_sub[0]
    zeta2[:, l1 - 1, l2 - 1, :] = zeta_list_sub[1]

    xnom_sub = np.load(
        os.path.join(
            LOG_DIRECTORY, FILE_NAME + '_' + str(l1) + str(l2) + '_xs.npy'
        )
    )
    xnom[:, l1 - 1, l2 - 1, :] = xnom_sub

# Constructs a Game-induced NOD and simulates it.
GiNOD_list = []

z = np.zeros((4, N + 1))

for k in range(N):
  Z1_k = Z1[:, :, :, :, k]
  Z2_k = Z2[:, :, :, :, k]
  zeta1_k = zeta1[:, :, :, k]
  zeta2_k = zeta2[:, :, :, k]
  xnom_k = xnom[:, :, :, k]
  z_bias = 1e-3 * np.ones((2,))

  if k == 0:
    znom1 = np.zeros((2,))
    znom2 = np.zeros((2,))
  else:
    znom1 = z[:2, k - 1]
    znom2 = z[2:, k - 1]

  GiNOD = NonlinearOpinionDynamicsTwoPlayer(
      dim_P1=2, dim_P2=2, x_indices_P1=np.array((0, 1, 2, 3)),
      x_indices_P2=np.array((4, 5, 6, 7)), z_indices_P1=np.array((8, 9)),
      z_indices_P2=np.array((10, 11)), Z_P1=Z1_k, Z_P2=Z2_k, zeta_P1=zeta1_k,
      zeta_P2=zeta2_k, x_ph_nom=xnom_k, znom_P1=znom1, znom_P2=znom2,
      z_P1_bias=z_bias, z_P2_bias=z_bias, damping=0.5, T=TIME_RES
  )

  x_joint_k = np.hstack((xnom_k[:, 1, 1], z[:, k]))

  z[:, k + 1] = z[:, k] + TIME_RES * GiNOD.cont_time_dyn(x_joint_k)

  GiNOD_list.append(GiNOD)

  print(z[:, k])
