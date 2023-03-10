"""
Util functions.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)

TODO:
  - Remove unused class
  - Rewrite comments
"""

from distutils.log import error
from multiprocessing.sharedctypes import Value
import dill
import yaml
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


class Logger(object):
  """
    Logging utility. The log is stored as a dictionary of lists.
    """

  def __init__(self, filename):
    """
        Constructor.

        :param filename: Where to save log.
        :type filename: string
        """
    self._filename = filename
    self._log = {}

    # Make sure we can open the file!
    fp = open(self._filename, "wb")
    fp.close()

  def log(self, field, value):
    """
        Add the given value to the specified field.

        :param field: which field to add to
        :type field: string
        :param value: value to add
        :type value: arbitrary
        """
    if field not in self._log:
      self._log[field] = [value]
    else:
      self._log[field].append(value)

  def dump(self):
    fp = open(self._filename, "wb")
    dill.dump(self._log, fp)
    fp.close()


class Visualizer(object):
  """
  Fancy visualization class
  """

  def __init__(
      self, position_indices, renderable_costs, player_linestyles,
      show_last_k=1, fade_old=False, plot_lims=None, figure_number=1
  ):
    """
      Construct from list of position indices and renderable cost functions.

      :param position_indices: list of tuples of position indices (1/player)
      :type position_indices: [(uint, uint)]
      :param renderable_costs: list of cost functions that support rendering
      :type renderable_costs: [Cost]
      :param player_linestyles: list of line styles (1 per player, e.g. ".-r")
      :type player_colors: [string]
      :param show_last_k: how many of last trajectories to plot (-1 shows all)
      :type show_last_k: int
      :param fade_old: flag for fading older trajectories
      :type fade_old: bool
      :param plot_lims: plot limits [xlim_low, xlim_high, ylim_low, ylim_high]
      :type plot_lims: [float, float, float, float]
      :param figure_number: which figure number to operate on
      :type figure_number: uint
      """
    self._position_indices = position_indices
    self._renderable_costs = renderable_costs
    self._player_linestyles = player_linestyles
    self._show_last_k = show_last_k
    self._fade_old = fade_old
    self._figure_number = figure_number
    self._plot_lims = plot_lims
    self._num_players = len(position_indices)

    # Store history as list of trajectories.
    # Each trajectory is a dictionary of lists of states and controls.
    self._iterations = []
    self._history = []

  def add_trajectory(self, iteration, traj):
    """
      Add a new trajectory to the history.

      :param iteration: which iteration is this
      :type iteration: uint
      :param traj: trajectory
      :type traj: {"xs": [np.array], "u1s": [np.array], "u2s": [np.array]}
      """
    self._iterations.append(iteration)
    self._history.append(traj)

  def plot(self):
    """ Plot everything. """
    plt.figure(self._figure_number)
    plt.rc("text", usetex=True)

    ax = plt.gca()
    ax.set_xlabel("$x(t)$")
    ax.set_ylabel("$y(t)$")

    if self._plot_lims is not None:
      ax.set_xlim(self._plot_lims[0], self._plot_lims[1])
      ax.set_ylim(self._plot_lims[2], self._plot_lims[3])

    ax.set_aspect("equal")

    # Render all costs.
    for cost in self._renderable_costs:
      cost.render(ax)

    # Plot the history of trajectories for each player.
    if self._show_last_k < 0 or self._show_last_k >= len(self._history):
      show_last_k = len(self._history)
    else:
      show_last_k = self._show_last_k

    plotted_iterations = []
    for kk in range(len(self._history) - show_last_k, len(self._history)):
      traj = self._history[kk]
      iteration = self._iterations[kk]
      plotted_iterations.append(iteration)

      alpha = 1.0
      if self._fade_old:
        alpha = 1.0 - float(len(self._history) - kk) / show_last_k

      for ii in range(self._num_players):
        x_idx, y_idx = self._position_indices[ii]
        if len(traj["xs"][0].shape) == 1:
          xs = traj["xs"][x_idx, :]
          ys = traj["xs"][y_idx, :]
        elif len(traj["xs"][0].shape) == 2:
          xs = [x[x_idx, 0] for x in traj["xs"]]
          ys = [x[y_idx, 0] for x in traj["xs"]]
        else:
          raise ValueError("Incorrect state vector dimension!")
        plt.plot(
            xs, ys, self._player_linestyles[ii],
            label="Player {}, iteration {}".format(ii, iteration), alpha=alpha,
            markersize=5
        )

    plt.title(
        "ILQ solver solution (iterations {}-{})".format(
            plotted_iterations[0], plotted_iterations[-1]
        )
    )

  def plot_controls(self, player_number):
    """ Plot control for both players. """
    plt.figure(self._figure_number + player_number)
    uis = "u%ds" % player_number
    plt.plot([ui[0, 0] for ui in self._history[-1][uis]], "*:r", label="u1")
    plt.plot([ui[1, 0] for ui in self._history[-1][uis]], "*:b", label="u2")
    plt.legend()
    plt.title("Controls for Player %d" % player_number)


class Plotter(object):
  """
    Plotting utility intended to be used with the Logger.
    """

  def __init__(self, py_filename, mat_filename):
    """
        Constructor.

        :param py_filename: Where to load python log.
        :param mat_filename: Where to load matlab log.
        :type filename: string
        """
    fp = open(py_filename, "rb")
    self._log = dill.load(fp)
    fp.close()

    self._matlab_log = scipy.io.loadmat(mat_filename)

  def plot_scalar_fields(self, fields, title="", xlabel="", ylabel=""):
    """
        Plot several scalar-valued fields over all time.

        :param fields: list of fields to plot
        :type fields: list of strings
        """
    plt.figure()
    for f in fields:
      plt.plot(self._log[f], linewidth=2, markersize=12, label=f)

    plt.legend()
    self._set_title_and_axis_labels(title, xlabel, ylabel)

  def show(self):
    plt.show()

  def _set_title_and_axis_labels(self, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

  def plot_player_costs(self):
    self.plot_scalar_fields(['total_cost1', 'total_cost2'],
                            title='player cost over time', xlabel='time (s)',
                            ylabel='cost')

    plt.savefig('player_costs.png')

  def plot_trajectories(self):

    # Plot trajectory from iLQG
    xs = np.array(self._log['xs'])
    xs = xs[-1, :, :, :]

    x1s = xs[:, 0].flatten()
    x2s = xs[:, 1].flatten()

    plt.figure()
    plt.plot(x1s, x2s, label="iLQG")

    hji_xs = self._matlab_log['traj']
    plt.plot(hji_xs[0, :], hji_xs[1, :], label='HJI')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Final trajectory')
    plt.savefig('trajectory.png')


class Struct:
  """
  Struct for managing parameters.
  """

  def __init__(self, data):
    for key, value in data.items():
      setattr(self, key, value)


def load_config(file_path):
  """
  Loads the config file.
  Args:
      file_path (string): path to the parameter file.
  Returns:
      Struct: parameters.
  """
  with open(file_path) as f:
    data = yaml.safe_load(f)
  config = Struct(data)
  return config


def simulate_opinion_ctrl(
    jnt_sys, opn_dyn, T_sim: float, x0: DeviceArray, us_opp: DeviceArray,
    goal_x: float, goal_y: float, v_const: float = 0.7, k_param: float = 1.0,
    beta: float = jnp.pi / 4.0, z_offset_magnitude: float = 10.0
) -> DeviceArray:
  """
  Simulates the system in closed-loop with the opinion-based controller in
  https://arxiv.org/pdf/2210.01642.pdf

  TODO: Opponent uses a state feedback policy.

  Args:
      jnt_dyn (DynamicalSystem): joint dynamics
      opn_dyn (DynamicalSystem): opinion dynamics
      T_sim (float): simulation horizon
      x0 (DeviceArray): (nx,) where nx is the dimension of the joint system (
        physical subsystems plus (a number of) opinion dynamics)
        For each opinion dynamics, their state := (z, u) where z is the
        opinion state and u is the attention parameter
      us_opp (DeviceArray): (nu, T_sim) opponent's control trajectory
      goal_x (float): x coordinate of goal
      goal_y (float): y coordinate of goal
      v_const (float): constant velocity of the ego
      k_param (float): steering controller parameter
      beta (float): steering controller parameter
      z_offset_magnitude (float): steering controller parameter

  Returns:
      DeviceArray: state trajectory (nx, T_sim + 1)
      DeviceArray: control trajectory (nu_ego, T_sim)
  """
  xs = jnp.zeros((jnt_sys._x_dim, T_sim + 1))
  xs = xs.at[:, 0].set(x0)
  us_ego = jnp.zeros((jnt_sys.u_dims[0], T_sim))
  for k in range(T_sim):
    # Computes opinion-based steering controller.
    w_k = opn_dyn.get_opinion_ctrl(
        xs[:, k], goal_x, goal_y, k_param, beta, z_offset_magnitude
    )
    u_ego_k = jnp.array((w_k, v_const))
    us_ego = us_ego.at[:, k].set(u_ego_k)
    us_k = [u_ego_k, us_opp[:, k]]

    # Computes the next state for the joint system.
    xs = xs.at[:, k + 1].set(jnt_sys.disc_time_dyn(xs[:, k], us_k))
  return xs, us_ego


def wrapPi(angle):
  # makes a number -pi to pi
  while angle <= -np.pi:
    angle += 2 * np.pi
  while angle > np.pi:
    angle -= 2 * np.pi
  return angle
