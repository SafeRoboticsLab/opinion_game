"""
Opinion Game: Two car toll station example.
"""

import os
import time
import numpy as np
import jax.numpy as jnp

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
SAVE_TRAJ = True

# Players' options
car_R_opn = 1
car_H_opn = 2

# Creates subsystem dynamics.
car_R = Bicycle4D(l_f=0.5, l_r=0.5)
car_H = Bicycle4D(l_f=0.5, l_r=0.5)
car_R_xyth_indices_in_product_state = (0, 1, 2, 3)
car_H_xyth_indices_in_product_state = (4, 5, 6, 7)

# Creates joint system dynamics.
jnt_sys = ProductMultiPlayerDynamicalSystem([car_R, car_H], T=TIME_RES)
x_dim = jnt_sys._x_dim

# Initializes states and iLQ policies.
car_R_px0 = 0.0
car_R_py0 = 0.0
car_R_theta0 = wrapPi(jnp.arctan2(-car_R_py0, -car_R_px0))
car_R_v0 = 8.0
car_R_x0 = jnp.array([car_R_px0, car_R_py0, car_R_theta0, car_R_v0])

car_H_px0 = 0.0
car_H_py0 = 7.0
car_H_theta0 = wrapPi(jnp.arctan2(-car_H_py0, -car_H_px0))
car_H_v0 = 8.0
car_H_x0 = jnp.array([car_H_px0, car_H_py0, car_H_theta0, car_H_v0])

jnt_x0 = jnp.concatenate([car_R_x0, car_H_x0], axis=0)

car_R_Ps = jnp.zeros((car_R._u_dim, jnt_sys._x_dim, HORIZON_STEPS))
car_H_Ps = jnp.zeros((car_H._u_dim, jnt_sys._x_dim, HORIZON_STEPS))

car_R_alphas = jnp.zeros((car_R._u_dim, HORIZON_STEPS))
car_H_alphas = jnp.zeros((car_H._u_dim, HORIZON_STEPS))

# Creates environment and defines costs.
ctrl_slack = config.CTRL_LIMIT_SLACK_MULTIPLIER

#   -> Car R
car_R_px_index = 0
car_R_py_index = 1
car_R_vel_index = 3

if car_R_opn == 1:
  car_R_goal_py = config.GOAL_PY_1
elif car_R_opn == 2:
  car_R_goal_py = config.GOAL_PY_2

car_R_goal_py_cost = ReferenceDeviationCost(
    reference=car_R_goal_py, dimension=car_R_py_index, is_x=True,
    name="car_R_goal_py", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)  # Tracks the target lane (y-position).

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

exit()

car_R_w_cost = QuadraticCost(
    0, 0.0, False, "car_R_w_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)
car_R_a_cost = QuadraticCost(
    1, 0.0, False, "car_R_a_cost", HORIZON_STEPS, x_dim, car_R._u_dim
)  # Control costs.

car_R_w_constr_cost = BoxInputConstraintCost(
    0, ctrl_slack * config.W_MIN, ctrl_slack * config.W_MAX, q1=1., q2=5.,
    name="car_R_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)

car_R_a_constr_cost = BoxInputConstraintCost(
    1, ctrl_slack * config.A_MIN, ctrl_slack * config.A_MAX, q1=1., q2=5.,
    name="car_R_a_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_R._u_dim
)

#   -> Car H
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
    0, ctrl_slack * config.W_MIN, ctrl_slack * config.W_MAX, q1=1., q2=5.,
    name="car_H_w_constr_cost", horizon=HORIZON_STEPS, x_dim=x_dim,
    ui_dim=car_H._u_dim
)

car_H_a_constr_cost = BoxInputConstraintCost(
    1, ctrl_slack * config.A_MIN, ctrl_slack * config.A_MAX, q1=1., q2=5.,
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
