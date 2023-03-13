"""
Multiplayer dynamical systems.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)

TODO:
  - Rewrite comments
"""
import torch
import numpy as np
from typing import Tuple
from scipy.linalg import block_diag

from functools import partial
from jax import jit, jacfwd
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp

from iLQGame.dynamical_system import Car4D


class MultiPlayerDynamicalSystem(object):
  """
  Base class for all multiplayer continuous-time dynamical systems. Supports
  numrical integration and linearization.
  """

  def __init__(self, x_dim, u_dims, T=0.1):
    """
    Initialize with number of state/control dimensions.

    :param x_dim: number of state dimensions
    :type x_dim: uint
    :param u_dims: liset of number of control dimensions for each player
    :type u_dims: [uint]
    :param T: time interval
    :type T: float
    """
    self._x_dim = x_dim
    self._u_dims = u_dims
    self._T = T
    self._num_players = len(u_dims)

    # Pre-computes Jacobian matrices.
    self.jac_f = jit(jacfwd(self.disc_time_dyn, argnums=[0, 1]))

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, u_list: list, *args) -> list:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): joint state (nx,)
        u_list (list of DeviceArray): list of controls [(nu_0,), (nu_1,), ...]

    Returns:
        list of DeviceArray: list of next states [(nx_0,), (nx_1,), ...]
    """
    raise NotImplementedError("cont_time_dyn() has not been implemented.")

  @partial(jit, static_argnums=(0,))
  def disc_time_dyn(self, x0: DeviceArray, u0_list: list, *args) -> list:
    """
    Computes the one-step evolution of the system in discrete time with Euler
    integration.

    Args:
        x0 (DeviceArray): joint state (nx,)
        u0_list (list of DeviceArray): list of controls [(nu_0,), (nu_1,), ...]

    Returns:
        list of DeviceArray: list of next states [(nx_0,), (nx_1,), ...]
    """
    x_dot = self.cont_time_dyn(x0, u0_list, args)
    return x0 + self._T * x_dot

  @partial(jit, static_argnums=(0,))
  def linearize_discrete_jitted(self, x0: DeviceArray, u0_list: list,
                                *args) -> Tuple[DeviceArray, list]:
    """
    Compute the Jacobian linearization of the dynamics for a particular
    state `x0` and control `u0`. Outputs `A` and `B` matrices of a
    discrete-time linear system:
          ``` x(k + 1) - x0 = A (x(k) - x0) + B (u(k) - u0) ```

    Args:
        x0 (DeviceArray): joint state (nx,)
        u0_list (list of DeviceArray): list of controls [(nu_0,), (nu_1,), ...]

    Returns:
        DeviceArray: the Jacobian of next state w.r.t. x0.
        list of DeviceArray: the Jacobian of next state w.r.t. u0_i.
    """
    A_disc, B_disc = self.jac_f(x0, u0_list, args)
    return A_disc, B_disc


class ProductMultiPlayerDynamicalSystem(MultiPlayerDynamicalSystem):

  def __init__(self, subsystems, T=0.1):
    """
    Implements a multiplayer dynamical system who's dynamics decompose into a
    Cartesian product of single-player dynamical systems.

    Initialize with a list of dynamical systems.

    :param subsystems: list of component (single-player) dynamical systems
    :type subsystems: [DynamicalSystem]
    :param T: time interval
    :type T: float
    """
    self._subsystems = subsystems
    self._x_dims = [sys._x_dim for sys in subsystems]

    x_dim = sum(self._x_dims)
    self._x_dim = x_dim
    u_dims = [sys._u_dim for sys in subsystems]
    self.u_dims = u_dims

    super(ProductMultiPlayerDynamicalSystem, self).__init__(x_dim, u_dims, T)

    self.update_lifting_matrices()
    self._num_opn_dyn = 0

  def update_lifting_matrices(self):
    """
    Updates the lifting matrices.
    """
    # Creates lifting matrices LMx_i for subsystem i such that LMx_i @ x = xi.
    _split_index = np.hstack((0, np.cumsum(np.asarray(self._x_dims))))
    self._LMx = [np.zeros((xi_dim, self._x_dim)) for xi_dim in self._x_dims]
    for i in range(len(self._x_dims)):
      self._LMx[i][:, _split_index[i]:_split_index[i + 1]] = np.eye(
          self._x_dims[i]
      )
      self._LMx[i] = jnp.asarray(self._LMx[i])

    # Creates lifting matrices LMu_i for subsystem i such that LMu_i @ u = ui.
    u_dims = self.u_dims
    u_dim = sum(u_dims)
    _split_index = np.hstack((0, np.cumsum(np.asarray(u_dims))))
    self._LMu = [np.zeros((ui_dim, u_dim)) for ui_dim in u_dims]
    for i in range(self._num_players):
      self._LMu[i][:, _split_index[i]:_split_index[i + 1]] = np.eye(u_dims[i])
      self._LMu[i] = jnp.asarray(self._LMu[i])

  def add_opinion_dyn(self, opn_dyns):
    """
    Append the physical subsystems with opinion dynamics, which do not have
    controls but *should* be affected by the physical states.
    """
    opn_dyns._start_index = self._x_dim  # starting index of the opn. states
    self._subsystems.append(opn_dyns)
    self._num_opn_dyn += 1

    self._x_dim += opn_dyns._x_dim
    self._x_dims.append(opn_dyns._x_dim)

    self.update_lifting_matrices()
    self._LMx += [
        jnp.eye(self._x_dim)
    ] * self._num_opn_dyn  # opn. dyns. take in the joint state

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, u_list: list, *args) -> list:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): joint state (nx,)
        u_list (list of DeviceArray): list of controls [(nu_0,), (nu_1,), ...]

    Returns:
        list of DeviceArray: list of next states [(nx_0,), (nx_1,), ...]
    """
    u_list += [None] * self._num_opn_dyn
    x_dot_list = [
        subsys.cont_time_dyn(LMx @ x, u0, args)
        for subsys, LMx, u0 in zip(self._subsystems, self._LMx, u_list)
    ]
    return jnp.concatenate(x_dot_list, axis=0)


class TwoPlayerUnicycle4D(MultiPlayerDynamicalSystem):
  """
  4D unicycle model with disturbance. Dynamics are as follows:
                            \dot x     = v cos theta + u21
                            \dot y     = v sin theta + u22
                            \dot theta = u11
                            \dot v     = u12
  """

  def __init__(self, T=0.1):
    super(TwoPlayerUnicycle4D, self).__init__(4, [2, 2], T)

  def __call__(self, x, u):
    """
    Compute the time derivative of state for a particular state/control.
    NOTE: `x`, and all `u` should be 2D (i.e. column vectors).

    :param x: current state
    :type x: torch.Tensor or np.array
    :param u: list of current control inputs for all each player
    :type u: [torch.Tensor] or [np.array]
    :return: current time derivative of state
    :rtype: torch.Tensor or np.array
    """
    assert len(u) == self._num_players

    if isinstance(x, np.ndarray):
      x_dot = np.zeros((self._x_dim, 1))
      cos = np.cos
      sin = np.sin
    else:
      x_dot = torch.zeros((self._x_dim, 1))
      cos = torch.cos
      sin = torch.sin

    x_dot[0, 0] = x[3, 0] * cos(x[2, 0]) + u[1][0, 0]
    x_dot[1, 0] = x[3, 0] * sin(x[2, 0]) + u[1][1, 0]
    x_dot[2, 0] = u[0][0, 0]
    x_dot[3, 0] = u[0][1, 0]
    return x_dot
