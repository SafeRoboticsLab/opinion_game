"""
Dynamical systems.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)

TODO:
  - Rewrite comments
  - Remove opinion dynamics
"""

from typing import Tuple

from functools import partial
from jax import jit
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
  def cont_time_dyn(
      self, x0: DeviceArray, u0: DeviceArray, k: int = 0, *args
  ) -> DeviceArray:
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
  def disc_time_dyn(
      self, x0: DeviceArray, u0: DeviceArray, k: int = 0, args=()
  ) -> DeviceArray:
    """
    Computes the one-step evolution of the system in discrete time with Euler
    integration.

    Args:
        x0 (DeviceArray): (nx,)
        u0 (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    x_dot = self.cont_time_dyn(x0, u0, k, args)
    return x0 + self._T * x_dot

  @partial(jit, static_argnums=(0,))
  def linearize_discrete_jitted(
      self, x0: DeviceArray, u0: DeviceArray, k: int = 0, args=()
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
    A_disc, B_disc = self.jac_f(x0, u0, k, args)
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
  def cont_time_dyn(
      self, x: DeviceArray, u: DeviceArray, k: int = 0, *args
  ) -> DeviceArray:
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
  def cont_time_dyn(
      self, x: DeviceArray, u: DeviceArray, k: int = 0, *args
  ) -> DeviceArray:
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
  def cont_time_dyn(
      self, x: DeviceArray, u: DeviceArray, k: int = 0, *args
  ) -> DeviceArray:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (DeviceArray): (nx,)
        u (DeviceArray): (nu,)

    Returns:
        DeviceArray: next state (nx,)
    """
    beta = jnp.arctan((self._l_r / (self._l_f + self._l_r)) * jnp.tan(u[1]))

    x0_dot = x[3] * jnp.cos(x[2] + beta)
    x1_dot = x[3] * jnp.sin(x[2] + beta)
    x2_dot = (x[3] / self._l_r) * jnp.sin(beta)
    x3_dot = u[0]

    return jnp.hstack((x0_dot, x1_dot, x2_dot, x3_dot))


class Car4D(DynamicalSystem):
  """
  4D car model. Dynamics are as follows
                          \dot x     = v cos theta
                          \dot y     = v sin theta
                          \dot theta = v * tan(u2) / l
                          \dot v     = u1
  """

  def __init__(self, l=3.0, T=0.1):
    self._l = l  # inter-axle length (m)
    super(Car4D, self).__init__(4, 2, T)

  @partial(jit, static_argnums=(0,))
  def cont_time_dyn(
      self, x: DeviceArray, u: DeviceArray, k: int = 0, *args
  ) -> DeviceArray:
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
    x2_dot = x[3] * jnp.tan(u[1]) / self._l
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
  def cont_time_dyn(
      self, x: DeviceArray, u: DeviceArray, k: int = 0, *args
  ) -> DeviceArray:
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
