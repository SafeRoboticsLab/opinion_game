"""
RHC test: Two car toll station example.
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
from planning.receding_horizon import RHCPlanner
from planning.utils import TwoCar8D

# Loads the config.
config = load_config("configs/example_two_car_L0_1.yaml")

# General parameters.
TIME_HORIZON = config.TIME_HORIZON  # planning horizon (s)
TIME_RES = config.TIME_RES  # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RES)
LOG_DIRECTORY = "experiments/two_car"
FILE_NAME = "two_car"

################################################################################
# Sets up subgames
################################################################################
# Creates subsystem dynamics.
car_R = Car4D(l=3.0, T=TIME_RES)
car_H = Car4D(l=3.0, T=TIME_RES)
car_R_xyth_indices_in_product_state = (0, 1, 2, 3)
car_H_xyth_indices_in_product_state = (4, 5, 6, 7)

TwoCar_casadi = TwoCar8D(l=3.0, T=TIME_RES)

# Creates joint system dynamics.
jnt_sys = ProductMultiPlayerDynamicalSystem([car_R, car_H], T=TIME_RES)
x_dim = jnt_sys._x_dim

# Defines costs.
#   -> Car R
car_R_px_index = 0
car_R_py_index = 1
car_R_psi_index = 2
car_R_vel_index = 3
car_R_position_indices_in_product_state = (0, 1)

car_R_goal_psi_cost = ReferenceDeviationCost(
    reference=0.0, dimension=car_R_psi_index, is_x=True, name="car_R_goal_psi",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)  # Tracks the target heading.

car_R_goal_vel_cost = ReferenceDeviationCost(
    reference=config.GOAL_VEL, dimension=car_R_vel_index, is_x=True,
    name="car_R_goal_vel", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)  # Tracks the target velocity.

car_R_maxv_cost = MaxVelCostPxDependent(
    v_index=car_R_vel_index, px_index=car_R_px_index, max_v=config.MAXV,
    px_lb=config.TOLL_STATION_PX_LB, px_ub=config.TOLL_STATION_PX_UB,
    name="car_R_maxv", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)  # Penalizes car speed above a threshold near the toll station.

car_R_lower_road_cost = SemiquadraticCost(
    dimension=car_R_py_index, threshold=config.ROAD_BOUNDARY_LOWER_THRESHOLD,
    oriented_right=False, is_x=True, name="car_R_lower_road_cost",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)
car_R_upper_road_cost = SemiquadraticCost(
    dimension=car_R_py_index, threshold=config.ROAD_BOUNDARY_UPPER_THRESHOLD,
    oriented_right=True, is_x=True, name="car_R_upper_road_cost",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)  # Round boundary costs.

car_R_min_vel_cost = SemiquadraticCost(
    dimension=car_R_vel_index, threshold=config.MINV, oriented_right=False,
    is_x=True, name="car_R_min_vel_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)  # Minimum velocity constraint.

car_R_a_cost = QuadraticCost(
    0, 0.0, False, "car_R_a_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)
car_R_w_cost = QuadraticCost(
    1, 0.0, False, "car_R_w_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)  # Control costs.

ctrl_slack = config.CTRL_LIMIT_SLACK_MULTIPLIER
car_R_a_constr_cost = BoxInputConstraintCost(
    0, ctrl_slack * config.A_MIN, ctrl_slack * config.A_MAX, q1=1., q2=5.,
    name="car_R_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)
car_R_w_constr_cost = BoxInputConstraintCost(
    1, ctrl_slack * config.W_MIN, ctrl_slack * config.W_MAX, q1=1., q2=5.,
    name="car_R_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)  # Control constraint costs.

#   -> Car H
car_H_px_index = 4
car_H_py_index = 5
car_H_psi_index = 6
car_H_vel_index = 7
car_H_position_indices_in_product_state = (4, 5)

car_H_goal_psi_cost = ReferenceDeviationCost(
    reference=0.0, dimension=car_H_psi_index, is_x=True, name="car_H_goal_psi",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)  # Tracks the target heading.

car_H_goal_vel_cost = ReferenceDeviationCost(
    reference=config.GOAL_VEL, dimension=car_H_vel_index, is_x=True,
    name="car_H_goal_vel", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)  # Tracks the target velocity.

car_H_maxv_cost = MaxVelCostPxDependent(
    v_index=car_H_vel_index, px_index=car_H_px_index, max_v=config.MAXV,
    px_lb=config.TOLL_STATION_PX_LB, px_ub=config.TOLL_STATION_PX_UB,
    name="car_H_maxv", horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)  # Penalizes car speed above a threshold near the toll station.

car_H_lower_road_cost = SemiquadraticCost(
    dimension=car_H_py_index, threshold=config.ROAD_BOUNDARY_LOWER_THRESHOLD,
    oriented_right=False, is_x=True, name="car_H_lower_road_cost",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)
car_H_upper_road_cost = SemiquadraticCost(
    dimension=car_H_py_index, threshold=config.ROAD_BOUNDARY_UPPER_THRESHOLD,
    oriented_right=True, is_x=True, name="car_H_upper_road_cost",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)  # Round boundary costs.

car_H_min_vel_cost = SemiquadraticCost(
    dimension=car_H_vel_index, threshold=config.MINV, oriented_right=False,
    is_x=True, name="car_H_min_vel_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)  # Minimum velocity constraint.

car_H_a_cost = QuadraticCost(
    0, 0.0, False, "car_H_a_cost", HORIZON_STEPS, x_dim, car_H._u_dim
)
car_H_w_cost = QuadraticCost(
    1, 0.0, False, "car_H_w_cost", HORIZON_STEPS, x_dim, car_H._u_dim
)  # Control costs.

car_H_a_constr_cost = BoxInputConstraintCost(
    0, ctrl_slack * config.A_MIN, ctrl_slack * config.A_MAX, q1=1., q2=5.,
    name="car_H_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)
car_H_w_constr_cost = BoxInputConstraintCost(
    1, ctrl_slack * config.W_MIN, ctrl_slack * config.W_MAX, q1=1., q2=5.,
    name="car_H_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)  # Control constraint costs.

# Proximity costs.
PROXIMITY_THRESHOLD = config.PROXIMITY_THRESHOLD
proximity_cost_RH = ProductStateProximityCostTwoPlayer([
    car_R_position_indices_in_product_state,
    car_H_position_indices_in_product_state,
], PROXIMITY_THRESHOLD, "proximity", HORIZON_STEPS, x_dim, car_R._u_dim)

# Build up total costs (intent-independent) for both players.
#   -> Robot
car_R_cost = PlayerCost()
car_R_cost.add_cost(car_R_goal_psi_cost, "x", 1.0)
car_R_cost.add_cost(car_R_goal_vel_cost, "x", 1.0)

car_R_cost.add_cost(car_R_lower_road_cost, "x", 50.0)
car_R_cost.add_cost(car_R_upper_road_cost, "x", 50.0)
car_R_cost.add_cost(car_R_min_vel_cost, "x", 50.0)
car_R_cost.add_cost(proximity_cost_RH, "x", 150.0)

car_R_player_id = 1
car_R_cost.add_cost(car_R_w_cost, car_R_player_id, 10.0)
car_R_cost.add_cost(car_R_a_cost, car_R_player_id, 1.0)

car_R_cost.add_cost(car_R_w_constr_cost, car_R_player_id, 50.0)
car_R_cost.add_cost(car_R_a_constr_cost, car_R_player_id, 50.0)

#   -> Human
car_H_cost = PlayerCost()
car_H_cost.add_cost(car_H_goal_psi_cost, "x", 1.0)
car_H_cost.add_cost(car_H_goal_vel_cost, "x", 1.0)

car_H_cost.add_cost(car_H_lower_road_cost, "x", 50.0)
car_H_cost.add_cost(car_H_upper_road_cost, "x", 50.0)
car_H_cost.add_cost(car_H_min_vel_cost, "x", 50.0)
car_H_cost.add_cost(proximity_cost_RH, "x", 150.0)

car_H_player_id = 2
car_H_cost.add_cost(car_H_w_cost, car_H_player_id, 10.0)
car_H_cost.add_cost(car_H_a_cost, car_H_player_id, 1.0)

car_H_cost.add_cost(car_H_w_constr_cost, car_H_player_id, 50.0)
car_H_cost.add_cost(car_H_a_constr_cost, car_H_player_id, 50.0)

# Toll station avoidance costs (multiple balls).
ts_px = config.TOLL_STATION_PX_LB
ts_py = config.TOLL_STATION_2_PY
while ts_px < config.TOLL_STATION_PX_UB:
  car_R_toll_station_cost_tmp = ProximityCost(
      position_indices=car_R_position_indices_in_product_state, point_px=ts_px,
      point_py=ts_py, max_distance=config.TOLL_STATION_WIDTH, name="",
      horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
  )
  car_R_cost.add_cost(car_R_toll_station_cost_tmp, "x", 150.0)

  car_H_toll_station_cost_tmp = ProximityCost(
      position_indices=car_H_position_indices_in_product_state, point_px=ts_px,
      point_py=ts_py, max_distance=config.TOLL_STATION_WIDTH, name="",
      horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
  )
  car_H_cost.add_cost(car_H_toll_station_cost_tmp, "x", 150.0)

  ts_px += config.TOLL_STATION_WIDTH

# Input constraints (for clipping).
a_min = config.A_MIN
a_max = config.A_MAX
w_min = config.W_MIN
w_max = config.W_MAX
u_constraints_car_R = BoxConstraint(
    lower=jnp.hstack((a_min, w_min)), upper=jnp.hstack((a_max, w_max))
)
u_constraints_car_H = BoxConstraint(
    lower=jnp.hstack((a_min, w_min)), upper=jnp.hstack((a_max, w_max))
)

# Initializes strategies.
car_R_Ps = jnp.zeros((car_R._u_dim, jnt_sys._x_dim, HORIZON_STEPS))
car_H_Ps = jnp.zeros((car_H._u_dim, jnt_sys._x_dim, HORIZON_STEPS))

car_R_alphas = jnp.zeros((car_R._u_dim, HORIZON_STEPS))
car_H_alphas = jnp.zeros((car_H._u_dim, HORIZON_STEPS))

# Sets up intent-dependent cost (Tracks the target lane (y-position)).
car_R_tgt_booth_cost_1 = ReferenceDeviationCostPxDependent(
    reference=config.GOAL_PY_1, dimension=car_R_py_index,
    px_dim=car_R_px_index, px_lb=config.GOAL_PX_LB,
    name="car_R_tgt_booth_cost_1", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)
car_R_tgt_booth_cost_2 = ReferenceDeviationCostPxDependent(
    reference=config.GOAL_PY_2, dimension=car_R_py_index,
    px_dim=car_R_px_index, px_lb=config.GOAL_PX_LB,
    name="car_R_tgt_booth_cost_2", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)
car_H_tgt_booth_cost_1 = ReferenceDeviationCostPxDependent(
    reference=config.GOAL_PY_1, dimension=car_H_py_index,
    px_dim=car_H_px_index, px_lb=config.GOAL_PX_LB,
    name="car_H_tgt_booth_cost_1", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)
car_H_tgt_booth_cost_2 = ReferenceDeviationCostPxDependent(
    reference=config.GOAL_PY_2, dimension=car_H_py_index,
    px_dim=car_H_px_index, px_lb=config.GOAL_PX_LB,
    name="car_H_tgt_booth_cost_2", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)

car_R_cost_subgame11 = deepcopy(car_R_cost)
car_H_cost_subgame11 = deepcopy(car_H_cost)
car_R_cost_subgame11.add_cost(car_R_tgt_booth_cost_1, "x", config.GOAL_W_P1_1)
car_H_cost_subgame11.add_cost(car_H_tgt_booth_cost_1, "x", config.GOAL_W_P2_1)

car_R_cost_subgame12 = deepcopy(car_R_cost)
car_H_cost_subgame12 = deepcopy(car_H_cost)
car_R_cost_subgame12.add_cost(car_R_tgt_booth_cost_1, "x", config.GOAL_W_P1_1)
car_H_cost_subgame12.add_cost(car_H_tgt_booth_cost_2, "x", config.GOAL_W_P2_2)

car_R_cost_subgame21 = deepcopy(car_R_cost)
car_H_cost_subgame21 = deepcopy(car_H_cost)
car_R_cost_subgame21.add_cost(car_R_tgt_booth_cost_2, "x", config.GOAL_W_P1_2)
car_H_cost_subgame21.add_cost(car_H_tgt_booth_cost_1, "x", config.GOAL_W_P2_1)

car_R_cost_subgame22 = deepcopy(car_R_cost)
car_H_cost_subgame22 = deepcopy(car_H_cost)
car_R_cost_subgame22.add_cost(car_R_tgt_booth_cost_2, "x", config.GOAL_W_P1_2)
car_H_cost_subgame22.add_cost(car_H_tgt_booth_cost_2, "x", config.GOAL_W_P2_2)

# Sets up ILQSolvers for all subgames.
alpha_scaling = np.linspace(0.01, 2.0, config.ALPHA_SCALING_NUM)
# alpha_scaling = np.logspace(-2, -0.04, config.ALPHA_SCALING_NUM)

solver11 = ILQSolver(
    jnt_sys, [car_R_cost_subgame11, car_H_cost_subgame11],
    [car_R_Ps, car_H_Ps], [car_R_alphas, car_H_alphas], alpha_scaling,
    config.MAX_ITER, u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=config.VERBOSE, name="subgame_11"
)
solver12 = ILQSolver(
    jnt_sys, [car_R_cost_subgame12, car_H_cost_subgame12],
    [car_R_Ps, car_H_Ps], [car_R_alphas, car_H_alphas], alpha_scaling,
    config.MAX_ITER, u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=config.VERBOSE, name="subgame_12"
)
solver21 = ILQSolver(
    jnt_sys, [car_R_cost_subgame21, car_H_cost_subgame21],
    [car_R_Ps, car_H_Ps], [car_R_alphas, car_H_alphas], alpha_scaling,
    config.MAX_ITER, u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=config.VERBOSE, name="subgame_21"
)
solver22 = ILQSolver(
    jnt_sys, [car_R_cost_subgame22, car_H_cost_subgame22],
    [car_R_Ps, car_H_Ps], [car_R_alphas, car_H_alphas], alpha_scaling,
    config.MAX_ITER, u_constraints=[u_constraints_car_R, u_constraints_car_H],
    verbose=config.VERBOSE, name="subgame_22"
)

subgames = [[solver11, solver12], [solver21, solver22]]

################################################################################
# RHC Planning
################################################################################
# Initializes states.
car_R_px0 = 0.0
car_R_py0 = 3.5
car_R_theta0 = 0.0
car_R_v0 = 3.0
car_R_x0 = np.array([car_R_px0, car_R_py0, car_R_theta0, car_R_v0])

car_H_px0 = 7.0
car_H_py0 = 3.5
car_H_theta0 = 0.0
car_H_v0 = 3.0
car_H_x0 = np.array([car_H_px0, car_H_py0, car_H_theta0, car_H_v0])

jnt_x0 = np.concatenate([car_R_x0, car_H_x0], axis=0)
z0 = np.array(([1e-3, 1e-3, 1e-3, 1e-3, 0., 0.]))

# Creates the GiNOD.
GiNOD = NonlinearOpinionDynamicsTwoPlayer(
    x_indices_P1=np.array((0, 1, 2, 3)),
    x_indices_P2=np.array((4, 5, 6, 7)),
    z_indices_P1=np.array((8, 9)),
    z_indices_P2=np.array((10, 11)),
    att_indices_P1=np.array((12,)),
    att_indices_P2=np.array((13,)),
    z_P1_bias=0. * np.ones((2,)),
    z_P2_bias=0. * np.ones((2,)),
    T=TIME_RES,
    damping_opn=0.1,
    damping_att=[0.7, 0.7],  # [0.7, 0.7]
    rho=[0.8, 0.8],  # [0.7, 0.7]
)

# RHC planning and simulation.
N_sim = config.N_SIM
W_R = np.diag([5.0, 5.0])
W_H = np.diag([5.0, 5.0])

planner = RHCPlanner(
    subgames, N_sim, jnt_sys, TwoCar_casadi, GiNOD, method='QMDPL0',
    config=config, W_ctrl=[W_R, W_H]
)
planner.plan(jnt_x0, z0)

np.save(os.path.join(LOG_DIRECTORY, FILE_NAME + '_L0_xs.npy'), planner.xs)
np.save(os.path.join(LOG_DIRECTORY, FILE_NAME + '_L0_zs.npy'), planner.zs)
np.save(os.path.join(LOG_DIRECTORY, FILE_NAME + '_L0_Hs.npy'), planner.Hs)
np.save(os.path.join(LOG_DIRECTORY, FILE_NAME + '_L0_PoI.npy'), planner.PoI)
