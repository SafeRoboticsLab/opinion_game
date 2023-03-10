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
x_nom = np.zeros((8, 2, 2, N))
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

    x_nom_sub = np.load(
        os.path.join(
            LOG_DIRECTORY, FILE_NAME + '_' + str(l1) + str(l2) + '_xs.npy'
        )
    )
    x_nom[:, l1 - 1, l2 - 1, :] = x_nom_sub

# Construct Game-induced NODs
k = 0
