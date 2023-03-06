"""
Script to run the human-robot corridor navigation example in
https://arxiv.org/pdf/2210.01642.pdf

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil)

TODOs:
  - Add convergence condition function
  - Robot: Add human opinion manipulation cost
  - Reproduce Charlotte's results
  - Receding horizon planning (check my previous Colab).
  - Add linesearch.
  - Redefine __call__() for all costs.
  - Replace all polyline references with reference state points.
  - In ilq_solver: Add missing terms: grad_u and cross-Hessians, i.e. c_ux.
  - Add back ReferenceDeviationCost.
  - Make cost render compatible with Jax.
"""

import os
import time
import numpy as np
import jax.numpy as jnp

from iLQGame_Nash.cost import *
from iLQGame_Nash.utils import *
from iLQGame_Nash.geometry import *
from iLQGame_Nash.constraint import *
from iLQGame_Nash.dynamical_system import *
from iLQGame_Nash.multiplayer_dynamical_system import *

from iLQGame_Nash.ilq_solver import ILQSolver
from iLQGame_Nash.player_cost import PlayerCost

# Loads the config and track file.
config = load_config("ex_corridor.yaml")

# General parameters.
TIME_HORIZON = config.TIME_HORIZON  # s
TIME_RES = config.TIME_RES  # s
HORIZON_STEPS = int(TIME_HORIZON / TIME_RES)
LOG_DIRECTORY = "experiments/ilqgame_Nash/logs/corridor/"

FILE_NAME = "example1"
SAVE_TRAJ = True

# Creates subsystem dynamics.
car_R = Unicycle4D()
car_H = Unicycle4D()

# Creates opinion dynamics.
car_R_xyth_indices_in_product_state = (0, 1, 2)
car_H_xyth_indices_in_product_state = (4, 5, 6)
# opn_dyn_car_R = OpinionDynamics2DCorridor(
#     bias=config.BIAS_ROBOT, R=config.R_ROBOT,
#     indices_ego=car_R_xyth_indices_in_product_state,
#     indices_opp=car_H_xyth_indices_in_product_state
# )
opn_dyn_car_R = OpinionDynamics2DCorridorWithAngle(
    bias=config.BIAS_ROBOT, d=config.D_ROBOT, m=config.M_ROBOT,
    R=config.R_ROBOT, indices_ego=car_R_xyth_indices_in_product_state,
    indices_opp=car_H_xyth_indices_in_product_state
)

opn_dyn_car_H = OpinionDynamics2DCorridorWithAngle(
    bias=config.BIAS_HUMAN, d=config.D_HUMAN, m=config.M_HUMAN,
    R=config.R_HUMAN, reverse_eta_opp=True,
    indices_ego=car_H_xyth_indices_in_product_state,
    indices_opp=car_R_xyth_indices_in_product_state
)

# Creates joint system dynamics.
# Do NOT initialize with any opinion dynamics, e.g.
#   jnt_sys = ProductMultiPlayerDynamicalSystem([car_R, car_H, opn_dyn_car_R])
jnt_sys = ProductMultiPlayerDynamicalSystem([car_R, car_H], T=TIME_RES)
jnt_sys.add_opinion_dyn(opn_dyn_car_R)
jnt_sys.add_opinion_dyn(opn_dyn_car_H)
x_dim = jnt_sys._x_dim

# Chooses initial states and set initial control laws to zero, such that
# we start with a situation that looks like this:
#
#                  (car 2)
#             |       X       |
#             |       :       |
#             |      \./      |
#             |               |
#             |               |
#             |               |
#             |               |
#             |               |
#             |               |
#             |       ^       |
#             |       :       |               (+x)
#             |       :       |                |
#             |       X       |                |
#                  (car 1)          (+y) ______|
#

# Initializes states and iLQ policies.
car_R_px0 = -2.0
car_R_py0 = 0.0
car_R_theta0 = wrapPi(jnp.arctan2(-car_R_py0, -car_R_px0))
car_R_v0 = 0.0
car_R_x0 = jnp.array([car_R_px0, car_R_py0, car_R_theta0, car_R_v0])

car_H_px0 = 2.0
car_H_py0 = 0.0
car_H_theta0 = wrapPi(jnp.arctan2(-car_H_py0, -car_H_px0))
car_H_v0 = 0.0
car_H_x0 = jnp.array([car_H_px0, car_H_py0, car_H_theta0, car_H_v0])

opn_z0 = 0.0
opn_u0 = 0.0
car_R_opn0 = jnp.array([opn_z0, opn_u0])
car_H_opn0 = jnp.array([opn_z0, opn_u0])

jnt_x0 = jnp.concatenate([car_R_x0, car_H_x0, car_R_opn0, car_H_opn0], axis=0)

car_R_Ps = jnp.zeros((car_R._u_dim, jnt_sys._x_dim, HORIZON_STEPS))
car_H_Ps = jnp.zeros((car_H._u_dim, jnt_sys._x_dim, HORIZON_STEPS))

car_R_alphas = jnp.zeros((car_R._u_dim, HORIZON_STEPS))
car_H_alphas = jnp.zeros((car_H._u_dim, HORIZON_STEPS))

# Creates environment and defines costs.
ctrl_lsm = config.CTRL_LIMIT_SLACK_MULTIPLIER

#   -> Car 1
car_R_position_indices_in_product_state = (0, 1)

car_R_goal = Point(car_H_px0, car_H_py0)
car_R_goal_reward = ProximityCostInfMaxDist(
    car_R_position_indices_in_product_state, car_R_goal, name="car_R_goal",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim, plot_color="r"
)  # Tracks the goal.

car_R_v_index_in_product_state = 3
car_R_maxv = config.CAR_R_MAXV
car_R_maxv_cost = SemiquadraticCost(
    car_R_v_index_in_product_state, car_R_maxv, True, True, "car_R_maxv",
    HORIZON_STEPS, x_dim, car_R._u_dim
)  # Penalizes speed above a threshold.

car_R_w_cost = QuadraticCost(
    0, 0.0, False, "car_R_w_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)
car_R_a_cost = QuadraticCost(
    1, 0.0, False, "car_R_a_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)  # Control costs.

car_R_w_constr_cost = BoxInputConstraintCost(
    0, ctrl_lsm * config.W_MIN, ctrl_lsm * config.W_MAX, q1=1., q2=5.,
    name="car_R_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)

car_R_a_constr_cost = BoxInputConstraintCost(
    1, ctrl_lsm * config.A_MIN, ctrl_lsm * config.A_MAX, q1=1., q2=5.,
    name="car_R_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)

#   -> Car 2
car_H_position_indices_in_product_state = (4, 5)

car_H_goal = Point(car_R_px0, car_R_py0)
car_H_goal_reward = ProximityCostInfMaxDist(
    car_H_position_indices_in_product_state, car_H_goal, name="car_H_goal",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim, plot_color="g"
)  # Tracks the goal.

car_H_v_index_in_product_state = 7
car_H_maxv = config.CAR_H_MAXV
car_H_maxv_cost = SemiquadraticCost(
    car_H_v_index_in_product_state, car_H_maxv, True, True, "car_H_maxv",
    HORIZON_STEPS, x_dim, car_H._u_dim
)  # Penalizes speed above a threshold.

car_H_w_cost = QuadraticCost(
    0, 0.0, False, "car_H_w_cost", HORIZON_STEPS, x_dim, car_H._u_dim
)
car_H_a_cost = QuadraticCost(
    1, 0.0, False, "car_H_a_cost", HORIZON_STEPS, x_dim, car_H._u_dim
)  # Control costs.

car_H_w_constr_cost = BoxInputConstraintCost(
    0, ctrl_lsm * config.W_MIN, ctrl_lsm * config.W_MAX, q1=1., q2=5.,
    name="car_H_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)

car_H_a_constr_cost = BoxInputConstraintCost(
    1, ctrl_lsm * config.A_MIN, ctrl_lsm * config.A_MAX, q1=1., q2=5.,
    name="car_H_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)

# Opinion-guided costs.
car_R_opn_reward = OpinionGuidedCostCorridorTwoPlayer(
    indices_ego=(0, 1), indices_opp=(4, 5), z_index=8,
    py_offset=config.PY_OFFSET_ROBOT, reverse_px=False, name="car_R_opn_cost",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_R._u_dim
)

car_H_opn_reward = OpinionGuidedCostCorridorTwoPlayer(
    indices_ego=(4, 5), indices_opp=(0, 1), z_index=10,
    py_offset=config.PY_OFFSET_HUMAN, reverse_px=True, name="car_H_opn_cost",
    horizon=HORIZON_STEPS, x_dim=x_dim, ui_dim=car_H._u_dim
)

# Proximity costs.
PROXIMITY_THRESHOLD = config.PROXIMITY_THRESHOLD
proximity_cost = ProductStateProximityCostTwoPlayer([
    car_R_position_indices_in_product_state,
    car_H_position_indices_in_product_state,
], PROXIMITY_THRESHOLD, "proximity", HORIZON_STEPS, x_dim, car_R._u_dim)

# Build up total costs for both players.
#   -> Robot
car_R_cost = PlayerCost()
car_R_cost.add_cost(car_R_goal_reward, "x", config.GOAL_REWARD_WEIGHT_ROBOT)
car_R_cost.add_cost(car_R_maxv_cost, "x", 5.0)
car_R_cost.add_cost(proximity_cost, "x", config.PROXIMITY_COST_WEIGHT_ROBOT)

car_R_player_id = 1
car_R_cost.add_cost(car_R_w_cost, car_R_player_id, 10.0)
car_R_cost.add_cost(car_R_a_cost, car_R_player_id, 1.0)

car_R_cost.add_cost(car_R_w_constr_cost, car_R_player_id, 1.0)
car_R_cost.add_cost(car_R_a_constr_cost, car_R_player_id, 1.0)

car_R_cost.add_cost(car_R_opn_reward, config.OPN_REWARD_WEIGHT_ROBOT)

#   -> Human
car_H_cost = PlayerCost()
car_H_cost.add_cost(car_H_goal_reward, "x", config.GOAL_REWARD_WEIGHT_HUMAN)
car_H_cost.add_cost(car_H_maxv_cost, "x", 5.0)
car_H_cost.add_cost(proximity_cost, "x", config.PROXIMITY_COST_WEIGHT_HUMAN)

car_H_player_id = 2
car_H_cost.add_cost(car_H_w_cost, car_H_player_id, 10.0)
car_H_cost.add_cost(car_H_a_cost, car_H_player_id, 1.0)

car_H_cost.add_cost(car_H_w_constr_cost, car_H_player_id, 1.0)
car_H_cost.add_cost(car_H_a_constr_cost, car_H_player_id, 1.0)

car_H_cost.add_cost(car_H_opn_reward, config.OPN_REWARD_WEIGHT_HUMAN)

# Input constraints (for clipping).
w_min = config.W_MIN
w_max = config.W_MAX
a_min = config.A_MIN
a_max = config.A_MAX
u_constraints_car_R = BoxConstraint(
    lower=jnp.hstack((w_min, a_min)), upper=jnp.hstack((w_max, a_max))
)
u_constraints_car_H = BoxConstraint(
    lower=jnp.hstack((w_min, a_min)), upper=jnp.hstack((w_max, a_max))
)

# Visualizer.
position_indices = [
    car_R_position_indices_in_product_state,
    car_H_position_indices_in_product_state,
]
renderable_costs = [car_R_goal_reward, car_H_goal_reward]

visualizer = Visualizer(
    position_indices, renderable_costs, [".-r", ".-g"], 1, False,
    plot_lims=[-5, 5, -2, 2]
)

# Logger.
if not os.path.exists(LOG_DIRECTORY):
  os.makedirs(LOG_DIRECTORY)

logger = Logger(os.path.join(LOG_DIRECTORY, 'corridor_example.pkl'))

# Set up ILQSolver.
solver = ILQSolver(
    jnt_sys, [car_R_cost, car_H_cost], jnt_x0, [car_R_Ps, car_H_Ps],
    [car_R_alphas, car_H_alphas], config.ALPHA_SCALING, config.MAX_ITER, None,
    logger, visualizer, [u_constraints_car_R, u_constraints_car_H]
)

# Runs iLQ.
solver.run()

# Plots results.
solver.plot_opinions(opn_state_idx_list=[[8, 9], [10, 11]])
# solver.plot_controls(player_id=car_R_player_id)
# solver.plot_controls(player_id=car_H_player_id)

# Runs opinion controller.
us_H = solver._current_operating_point[1][1]
xs, us_R = simulate_opinion_ctrl(
    jnt_sys, opn_dyn_car_R, T_sim=HORIZON_STEPS, x0=jnt_x0, us_opp=us_H,
    goal_x=car_H_px0, goal_y=car_H_py0, v_const=0.95, k_param=1.0,
    beta=jnp.pi / 4.0, z_offset_magnitude=10.0
)

print(xs, "\n", us_R)
traj_opn_ctrl = {"xs": np.asarray(xs)}
traj_opn_ctrl["u%ds" % 1] = np.asarray(us_R)
traj_opn_ctrl["u%ds" % 2] = np.asarray(us_H)
visualizer.add_trajectory(iteration=0, traj=traj_opn_ctrl)
plt.clf()
visualizer.plot()
plt.show()

# Saves results.
if SAVE_TRAJ:
  np.save(
      os.path.join(LOG_DIRECTORY, FILE_NAME + '_opn_game_xs.npy'),
      solver._current_operating_point[0]
  )

  np.save(
      os.path.join(LOG_DIRECTORY, FILE_NAME + '_baseline_xs.npy'),
      np.asarray(xs)
  )
