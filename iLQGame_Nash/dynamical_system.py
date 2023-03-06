"""
Dynamical systems.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)
"""

from typing import Tuple

from functools import partial
from jax import jit, lax
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


class DynamicalSystem(object):
  """
  Base class for all continuous-time dynamical systems. Supports numerical
  integration and linearization.
  """

  def __init__(self, x_dim, u_dim, T=0.1):
    """
    Initialize with number of state/control dimensions.

    :param x_dim: number of state dimensions
    :type x_dim: uint
    :param u_dim: number of control dimensions
    :type u_dim: uint
    :param T: time interval
    :type T: float
    """
    self._x_dim = x_dim
    self._u_dim = u_dim
    self._T = T

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x0: DeviceArray, u0: DeviceArray) -> DeviceArray:
    """
    Abstract method.
    Computes the time derivative of state for a particular state/control.

    Args:
        x0 (DeviceArray): (nx,)
        u0 (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    raise NotImplementedError("cont_time_dyn() has not been implemented.")

  @partial(jit, static_argnums=(0,))
  def disc_time_dyn(self, x0: DeviceArray, u0: DeviceArray) -> DeviceArray:
    """
    Computes the one-step evolution of the system in discrete time with Euler
    integration.

    Args:
        x0 (DeviceArray): (nx,)
        u0 (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    x_dot = self.cont_time_dyn(x0, u0)
    return x0 + self._T * x_dot

  @partial(jit, static_argnums=(0,))
  def linearize_discrete_jitted(
      self, x0: DeviceArray, u0: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """
    Compute the Jacobian linearization of the dynamics for a particular
    state `x0` and control `u0`. Outputs `A` and `B` matrices of a
    discrete-time linear system:
          ``` x(k + 1) - x0 = A (x(k) - x0) + B (u(k) - u0) ```

    Args:
        x0 (DeviceArray): (nx,)
        u0 (DeviceArray): (nu,)

    Returns:
        DeviceArray: the Jacobian of next state w.r.t. the current state.
        DeviceArray: the Jacobian of next state w.r.t. the current control.
    """
    A_disc, B_disc = self.jac_f(x0, u0)
    return A_disc, B_disc


class Unicycle4D(DynamicalSystem):
  """
  4D unicycle model. Dynamics are as follows:
                            \dot x     = v cos theta
                            \dot y     = v sin theta
                            \dot theta = u1
                            \dot v     = u2
  """

  def __init__(self, T=0.1):
    super(Unicycle4D, self).__init__(4, 2, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, u: DeviceArray) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): (nx,)
        u (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    x0_dot = x[3] * jnp.cos(x[2])
    x1_dot = x[3] * jnp.sin(x[2])
    x2_dot = u[0]
    x3_dot = u[1]
    return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot))


class PointMass2D(DynamicalSystem):
  """
  2D unicycle model (which actually has 4D state). Dynamics are as follows:
                          \dot x  = vx
                          \dot y  = vy
                          \dot vx = u1
                          \dot vy = u2
  """

  def __init__(self, T=0.1):
    super(PointMass2D, self).__init__(4, 2, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, u: DeviceArray) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): (nx,)
        u (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    x0_dot = x[2]
    x1_dot = x[3]
    x2_dot = u[0]
    x3_dot = u[1]
    return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot))


class Bicycle4D(DynamicalSystem):
  """
  4D (kinematic) bicycle model. Dynamics are as follows:
                          \dot x     = v cos(psi + beta)
                          \dot y     = v sin(psi + beta)
                          \dot psi   = (v / l_r) sin(beta)
                          \dot v     = u1
                  where beta = arctan((l_r / (l_f + l_r)) tan(u2))

  Dynamics were taken from:
  https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf

  `psi` is the inertial heading.
  `beta` is the angle of the current velocity of the center of mass with respect
      to the longitudinal axis of the car
  `u1` is the acceleration of the center of mass in the same direction as the
      velocity.
  `u2` is the front steering angle.
  """

  def __init__(self, l_f, l_r, T=0.1):
    """
    Initialize with front and rear lengths.

    :param l_f: distance (m) between center of mass and front axle
    :type l_f: float
    :param l_r: distance (m) between center of mass and rear axle
    :type l_r: float
    """
    self._l_f = l_f
    self._l_r = l_r
    super(Bicycle4D, self).__init__(4, 2, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, u: DeviceArray) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): (nx,)
        u (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    beta = jnp.atan((self._l_r / (self._l_f + self._l_r)) * jnp.tan(u[1]))

    x0_dot = x[3] * jnp.cos(x[2] + beta)
    x1_dot = x[3] * jnp.sin(x[2] + beta)
    x2_dot = (x[3] / self._l_r) * jnp.sin(beta)
    x3_dot = u[0]

    return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot))


class Car5D(DynamicalSystem):
  """
  5D car model. Dynamics are as follows, adapted from
  https://ac.els-cdn.com/S2405896316301215/1-s2.0-S2405896316301215-main.pdf?_ti
  d=ad143a13-6571-4733-a984-1b5a41960e78&acdnat=1552430727_12aedd0da2ca11eb07eef
  49d27b5ab12
                          \dot x     = v cos theta
                          \dot y     = v sin theta
                          \dot theta = v * tan(phi) / l
                          \dot phi   = u1
                          \dot v     = u2
  """

  def __init__(self, l=3.0, T=0.1):
    self._l = l  # inter-axle length (m)
    super(Car5D, self).__init__(5, 2, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, u: DeviceArray) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): (nx,)
        u (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    x0_dot = x[4] * jnp.cos(x[2])
    x1_dot = x[4] * jnp.sin(x[2])
    x2_dot = x[4] * jnp.tan(x[3]) / self._l
    x3_dot = u[0]
    x4_dot = u[1]
    return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot, x4_dot))


class OpinionDynamics2DCorridor(DynamicalSystem):
  """
  2D opinion dynamics model. Dynamics are as follows, adapted from
  https://arxiv.org/pdf/2210.01642.pdf (Eqn. (2a) & (2b))
      \dot z = -d z + u tanh(\alpha z + bias)
      \dot u = -m u + exp(c (R - ||pos_{ego} - pos_{oppo}||))
      where
      pos := (px, py)
      and
      bias = 0 (unbiased), > 0 (bias left), < 0 (bias right)
  """

  def __init__(
      self, bias, indices_ego, indices_opp, d=0.5, alpha=0.1, m=1.0, c=1.0,
      R=9.0, sqrt_eps=1e-6, exp_thresh=10, T=0.1, start_index=None
  ):
    """
    Initializer, dependent on the physical subsystems.

    Args:
        bias (float): bias = 0 (unbiased), > 0 (bias left), < 0 (bias right)
        indices_ego (tuple): (px_index, py_index, theta_index) in joint system
        indices_opp (tuple): (px_index, py_index, theta_index) in joint system
        d (float, optional): resistance parameter (> 0). Defaults to 0.5.
        alpha (float, optional): parameter (> 0). Defaults to 0.1.
        m (float, optional): parameter (> 0). Defaults to 1.0.
        c (float, optional): parameter (> 0). Defaults to 1.0.
        R (float, optional): attention radius (> 0). Defaults to 9.0.
        sqrt_eps (float, optional): small value for sqrt. Defaults to 1e-6.
        exp_thresh (float, optional): threshold of the exponential in u dynamics
        T (float, optional): time interval. Defaults to 0.1.
        start_index (int): index of the first state of opinion dynamics in
          joint system
    """
    self.x_dim = 2
    self.u_dim = 0
    self._bias = bias
    self._indices_ego = indices_ego
    self._indices_opp = indices_opp
    self._d = d
    self._alpha = alpha
    self._m = m
    self._c = c
    self._R = R
    self._sqrt_eps = sqrt_eps
    self._exp_thresh = exp_thresh
    self._start_index = start_index
    super(OpinionDynamics2DCorridor, self).__init__(2, 0, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, ctrl=None) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.
    This is an autonomous system.

    Args:
        x (DeviceArray): (nx,) where nx is the dimension of the joint system (
          physical subsystems plus (a number of) opinion dynamics)
          For each opinion dynamics, their state := (z, u) where z is the
          opinion state and u is the attention parameter
        ctrl (DeviceArray): None

    Returns:
        DeviceArray: next state (nx,)
    """
    _z_idx = self._start_index
    _u_idx = self._start_index + 1

    px_ego_index, py_ego_index, _ = self._indices_ego
    px_opp_index, py_opp_index, _ = self._indices_opp

    _px_ego = x[px_ego_index]
    _py_ego = x[py_ego_index]
    _px_opp = x[px_opp_index]
    _py_opp = x[py_opp_index]
    _dist = jnp.sqrt(
        jnp.maximum((_px_ego - _px_opp)**2 + (_py_ego - _py_opp)**2,
                    self._sqrt_eps)
    )

    z_now = x[_z_idx]
    u_now = x[_u_idx]

    z_dot = -self._d * z_now + u_now * jnp.tanh(
        self._alpha * z_now + self._bias
    )
    u_dot = -self._m * u_now + jnp.exp(
        jnp.minimum(self._c * (self._R - _dist), self._exp_thresh)
    )
    return jnp.hstack((z_dot, u_dot))


class OpinionDynamics2DCorridorWithAngle(DynamicalSystem):
  """
  2D opinion dynamics model. Dynamics are as follows, adapted from
  https://arxiv.org/pdf/2210.01642.pdf (Eqn. (2a) & (2b))
      \dot z = -d z + u tanh(\alpha z + gamma tan(\eta_{oppo}) + bias)
      \dot u = -m u + exp(c (R - ||pos_{ego} - pos_{oppo}||) cos(\eta_{ego}))
      where
      pos := (px, py)
      and
      bias = 0 (unbiased), > 0 (bias left), < 0 (bias right)
  """

  def __init__(
      self, bias, indices_ego, indices_opp, d=0.5, alpha=0.1, gamma=3.0, m=1.0,
      c=1.0, R=9.0, sqrt_eps=1e-6, exp_thresh=10, T=0.1, reverse_eta_opp=False,
      start_index=None
  ):
    """
    Initializer, dependent on the physical subsystems.

    Args:
        bias (float): bias = 0 (unbiased), > 0 (bias left), < 0 (bias right)
        indices_ego (tuple): (px_index, py_index, theta_index) in joint system
        indices_opp (tuple): (px_index, py_index, theta_index) in joint system
        d (float, optional): resistance parameter (> 0). Defaults to 0.5.
        alpha (float, optional): parameter (> 0). Defaults to 0.1.
        gamma (float, optional): parameter (\in R). Defaults to 3.0.
        m (float, optional): parameter (> 0). Defaults to 1.0.
        c (float, optional): parameter (> 0). Defaults to 1.0.
        R (float, optional): attention radius (> 0). Defaults to 9.0.
        sqrt_eps (float, optional): small value for sqrt. Defaults to 1e-6.
        exp_thresh (float, optional): threshold of the exponential in u dynamics
        T (float, optional): time interval. Defaults to 0.1.
        reverse_eta_opp (bool, optional): if reversing the sign of eta_opp.
        start_index (int): index of the first state of opinion dynamics in
          joint system
    """
    self.x_dim = 2
    self.u_dim = 0
    self._bias = bias
    self._indices_ego = indices_ego
    self._indices_opp = indices_opp
    self._d = d
    self._alpha = alpha
    self._gamma = gamma
    self._m = m
    self._c = c
    self._R = R
    self._sqrt_eps = sqrt_eps
    self._exp_thresh = exp_thresh
    if reverse_eta_opp:
      self._eta_opp_sign = -1.0
      self._px_sign = -1.0
    else:
      self._eta_opp_sign = 1.0
      self._px_sign = 1.0
    self._start_index = start_index
    super(OpinionDynamics2DCorridorWithAngle, self).__init__(2, 0, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(self, x: DeviceArray, ctrl=None) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.
    This is an autonomous system.

    Args:
        x (DeviceArray): (nx,) where nx is the dimension of the joint system (
          physical subsystems plus (a number of) opinion dynamics)
          For each opinion dynamics, their state := (z, u) where z is the
          opinion state and u is the attention parameter
        ctrl (DeviceArray): None

    Returns:
        DeviceArray: next state (nx,)
    """
    _z_idx = self._start_index
    _u_idx = self._start_index + 1

    px_ego_index, py_ego_index, theta_ego_index = self._indices_ego
    px_opp_index, py_opp_index, theta_opp_index = self._indices_opp

    _px_ego = x[px_ego_index]
    _py_ego = x[py_ego_index]
    _theta_ego = x[theta_ego_index]
    _px_opp = x[px_opp_index]
    _py_opp = x[py_opp_index]
    _theta_opp = x[theta_opp_index]

    _dist = jnp.sqrt(
        jnp.maximum((_px_ego - _px_opp)**2 + (_py_ego - _py_opp)**2,
                    self._sqrt_eps)
    )

    # NOTE: jnp.arctan2(y-coordinate, x-coordinate)

    _angle_ego_to_opp = jnp.arctan2(_py_ego - _py_opp, _px_ego - _px_opp)
    _eta_opp = self._eta_opp_sign * (_angle_ego_to_opp-_theta_opp)

    _angle_opp_to_ego = jnp.arctan2(_py_opp - _py_ego, _px_opp - _px_ego)
    _eta_ego = _angle_opp_to_ego - _theta_ego

    z_now = x[_z_idx]
    u_now = x[_u_idx]

    z_dot = -self._d * z_now + u_now * jnp.tanh(
        self._alpha * z_now + self._gamma * jnp.tan(_eta_opp) + self._bias
    )
    u_dot = -self._m * u_now + jnp.exp(
        jnp.minimum(
            self._c * jnp.cos(_eta_ego) * (self._R - _dist), self._exp_thresh
        )
    )
    return jnp.hstack((z_dot, u_dot))

  @partial(jit, static_argnums=(0,))
  def get_opinion_ctrl(
      self, x0: DeviceArray, goal_x: float, goal_y: float, k: float = 1.0,
      beta: float = jnp.pi / 4.0, z_offset_magnitude: float = 10.0
  ) -> DeviceArray:
    """
    Computes the opinion-based steering controller in
    https://arxiv.org/pdf/2210.01642.pdf

    Args:
        x0 (DeviceArray): (nx,) where nx is the dimension of the joint system (
          physical subsystems plus (a number of) opinion dynamics)
          For each opinion dynamics, their state := (z, u) where z is the
          opinion state and u is the attention parameter
        goal_x (float): x coordinate of goal
        goal_y (float): y coordinate of goal
        k (float): steering controller parameter
        beta (float): steering controller parameter
        z_offset_magnitude (float): steering controller parameter

    Returns:
        DeviceArray: next state (nx,)
    """

    def true_fn():
      # Ego has already passed the opponent.
      return self._px_sign * z_offset_magnitude

    def false_fn():
      return self._px_sign * z_offset_magnitude / 25.0

    _z_idx = self._start_index

    px_ego_index, py_ego_index, theta_ego_index = self._indices_ego
    px_opp_index, _, _ = self._indices_opp

    _px_ego = x0[px_ego_index]
    _py_ego = x0[py_ego_index]
    _theta_ego = x0[theta_ego_index]
    _px_opp = x0[px_opp_index]

    z_offset = lax.cond(
        self._px_sign * _px_ego > self._px_sign * _px_opp, true_fn, false_fn
    )

    z_now = x0[_z_idx]
    phi = jnp.arctan2(goal_y - _py_ego, goal_x - _px_ego) - _theta_ego
    steering_rate = k * jnp.sin(beta * jnp.tanh(z_now - z_offset) + phi)

    return steering_rate
