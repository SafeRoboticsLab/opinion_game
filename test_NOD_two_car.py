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

from opinion_dynamics.opinion_dynamics import NonlinearOpinionDynamicsTwoPlayer

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

# ------ Dynamic attention ------
z = np.zeros((6, N + 1))
Hs = np.zeros((4, 4, N))
PoI = np.zeros((2, N))

car_R_opn = 2
car_H_opn = 2

z_bias = 1e-3 * np.ones((2,))

GiNOD = NonlinearOpinionDynamicsTwoPlayer(
    x_indices_P1=np.array((0, 1, 2, 3)),
    x_indices_P2=np.array((4, 5, 6, 7)),
    z_indices_P1=np.array((8, 9)),
    z_indices_P2=np.array((10, 11)),
    att_indices_P1=np.array((12,)),
    att_indices_P2=np.array((13,)),
    z_P1_bias=z_bias,
    z_P2_bias=z_bias,
    T=TIME_RES,
    damping_opn=0.0,
    damping_att=1.0,
    rho=0.7,
)

for k in range(N):
  Z1_k = Z1[:, :, :, :, k]
  Z2_k = Z2[:, :, :, :, k]
  zeta1_k = zeta1[:, :, :, k]
  zeta2_k = zeta2[:, :, :, k]
  xnom_k = xnom[:, :, :, k]

  if k == 0:
    znom1_k = np.zeros((2,))
    znom2_k = np.zeros((2,))
  else:
    znom1_k = z[:2, k - 1]
    znom2_k = z[2:4, k - 1]

  x_joint_k = np.hstack((xnom_k[:, car_R_opn - 1, car_H_opn - 1], z[:, k]))

  z_dot_k, H_k, PoI1_k, PoI2_k = GiNOD.cont_time_dyn(
      x_joint_k, None, Z1_k, Z2_k, zeta1_k, zeta2_k, xnom_k, znom1_k, znom2_k
  )

  z[:, k + 1] = z[:, k] + TIME_RES*z_dot_k
  Hs[:, :, k] = H_k
  PoI[:, k] = np.array((PoI1_k, PoI2_k))

  print(z[:, k])
  # print(PoI1_k, PoI2_k)

exit()

np.save(
    os.path.join(
        LOG_DIRECTORY,
        FILE_NAME + '_' + str(car_R_opn) + str(car_H_opn) + '_opn.npy'
    ), z
)

np.save(
    os.path.join(
        LOG_DIRECTORY,
        FILE_NAME + '_' + str(car_R_opn) + str(car_H_opn) + '_Hs.npy'
    ), Hs
)

# # ------ Fixed attention ------
# # Constructs a Game-induced NOD and simulates it along subgame trajectories.
# Hs = np.zeros((4, 4, N + 1))

# z = np.zeros((4, N + 1))

# car_R_opn = 2
# car_H_opn = 2

# z_bias = 1e-3 * np.ones((2,))

# GiNOD = NonlinearOpinionDynamicsTwoPlayer(
#     x_indices_P1=np.array((0, 1, 2, 3)),
#     x_indices_P2=np.array((4, 5, 6, 7)),
#     z_indices_P1=np.array((8, 9)),
#     z_indices_P2=np.array((10, 11)),
#     att_indices_P1=np.array(()),
#     att_indices_P2=np.array(()),
#     z_P1_bias=z_bias,
#     z_P2_bias=z_bias,
#     T=TIME_RES,
#     damping_opn=0.0,
# )

# for k in range(N):
#   Z1_k = Z1[:, :, :, :, k]
#   Z2_k = Z2[:, :, :, :, k]
#   zeta1_k = zeta1[:, :, :, k]
#   zeta2_k = zeta2[:, :, :, k]
#   xnom_k = xnom[:, :, :, k]

#   if k == 0:
#     znom1_k = np.zeros((2,))
#     znom2_k = np.zeros((2,))
#   else:
#     znom1_k = z[:2, k - 1]
#     znom2_k = z[2:, k - 1]

#   x_joint_k = np.hstack((xnom_k[:, car_R_opn - 1, car_H_opn - 1], z[:, k]))

#   z_dot_k, H_k = GiNOD.cont_time_dyn_fixed_att(
#       x_joint_k, None, 2.0, 2.0, Z1_k, Z2_k, zeta1_k, zeta2_k, xnom_k, znom1_k,
#       znom2_k
#   )

#   z[:, k + 1] = z[:, k] + TIME_RES*z_dot_k

#   Hs[:, :, k] = H_k

#   print(z[:, k])

# np.save(
#     os.path.join(
#         LOG_DIRECTORY,
#         FILE_NAME + '_' + str(car_R_opn) + str(car_H_opn) + '_opn.npy'
#     ), z
# )

# np.save(
#     os.path.join(
#         LOG_DIRECTORY,
#         FILE_NAME + '_' + str(car_R_opn) + str(car_H_opn) + '_Hs.npy'
#     ), Hs
# )
