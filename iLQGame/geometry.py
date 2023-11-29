"""
Geometry objects for planning.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)
"""

import numpy as np

from functools import partial
from jax import jit, lax
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
import jax


class Point(object):
  """
  Point class for 2D points.
  """

  def __init__(self, x=0.0, y=0.0):
    self.x = x
    self.y = y

  def __add__(self, rhs):
    return Point(self.x + rhs.x, self.y + rhs.y)

  def __sub__(self, rhs):
    return Point(self.x - rhs.x, self.y - rhs.y)

  def __mul__(self, rhs):
    return Point(self.x * rhs, self.y * rhs)

  def __rmul__(self, lhs):
    return Point(self.x * lhs, self.y * lhs)

  def __imul__(self, rhs):
    self.x *= rhs
    self.y *= rhs

  def __truediv__(self, rhs):
    return Point(self.x / rhs, self.y / rhs)

  def __idiv__(self, rhs):
    self.x /= rhs
    self.y /= rhs

  def norm_squared(self):
    return self.x**2 + self.y**2

    return np.sqrt(self.norm_squared())


class LineSegment(object):
  """
  Class for 2D line segments.
  """

  def __init__(self, p1, p2):
    self.p1 = p1
    self.p2 = p2

  def __len__(self):
    return (self.p1 - self.p2).norm()

  def signed_distance_to(self, point):
    """
    Compute signed distance to other point.
    Sign convention is positive to the right and negative to the left, e.g.:
                                    *
                                    |
                negative            |             positive
                                    |
                                    |
                                    *

    Args:
        point (Point): query point
    """
    # Vector from p1 to query.
    relative = point - self.p1

    # Compute the unit direction of this line segment.
    direction = self.p2 - self.p1
    direction /= direction.norm()

    # Find signed length of projection and of cross product.
    projection = relative.x * direction.x + relative.y * direction.y
    cross = relative.x * direction.y - direction.x * relative.y
    cross_sign = 1.0 if cross >= 0.0 else -1.0

    if projection < 0.0:
      # Query lies behind this line segment, so closest distance will be
      # from p1.
      return cross_sign * relative.norm()
    elif projection > self.__len__():
      # Closest distance will be to p2.
      return cross_sign * (self.p2 - point).norm()
    else:
      return cross


class Polyline(object):
  """
  Polyline class to represent piecewise linear path in 2D.
  """

  def __init__(self, points=[]):
    """
    Initialize from a list of points. Keeps only a reference to input list.

    Args:
        points ([Point]): list of Points
    """
    self.points = points

  def signed_distance_to(self, point):
    """
    Compute signed distance from this polyline to the given point.
    Sign convention is positive to the right and negative to the left, e.g.:
                                    *
                                    |
                negative            |             positive
                                    |
                                    |
                                    *

    Args:
        point (Point): query point
    """
    # NOTE: for now, we'll just implement this with a naive linear search.
    # In future, if we want to optimize at all we can pass in a guess of
    # which index we expect the closest point to be close to.
    best_signed_distance = float("inf")
    for ii in range(1, len(self.points)):
      segment = LineSegment(self.points[ii - 1], self.points[ii])
      signed_distance = segment.signed_distance_to(point)

      if abs(signed_distance) < abs(best_signed_distance):
        best_signed_distance = signed_distance

    return best_signed_distance


# ---------------------------- Jitted functions ------------------------------
class LineSegment_jitted(object):
  """
  Class for 2D line segments.
  """

  def __init__(self, p1: ArrayImpl, p2: ArrayImpl):
    """
    Initialization.

    Args:
        p1 (ArrayImpl): px and py (2,)
        p2 (ArrayImpl): px and py (2,)
    """
    self.p1 = p1
    self.p2 = p2
    self.length = np.linalg.norm(self.p1 - self.p2)

  @partial(jit, static_argnums=(0,))
  def signed_distance_to(self, point: ArrayImpl) -> ArrayImpl:
    """
    Computes signed distance to other point.
    Sign convention is positive to the right and negative to the left, e.g.:
                                    *
                                    |
                negative            |             positive
                                    |
                                    |
                                    *

    Args:
        point (ArrayImpl): query point (2,)

    Returns:
        ArrayImpl: scalar
    """

    def true_fn_outer(projection, cross, dist_p1, dist_p2):
      return dist_p1

    def false_fn_outer(projection, cross, dist_p1, dist_p2):

      def true_fn_inner(projection, cross, dist_p1, dist_p2):
        return dist_p2

      def false_fn_inner(projection, cross, dist_p1, dist_p2):
        return cross

      return lax.cond(
          projection > self.length, true_fn_inner, false_fn_inner, projection, cross, dist_p1,
          dist_p2
      )

    # Vector from p1 to query.
    relative = point - self.p1

    # Compute the unit direction of this line segment.
    direction = self.p2 - self.p1
    direction /= jnp.linalg.norm(direction)

    # Find signed length of projection and of cross product.
    projection = relative[0] * direction[0] + relative[1] * direction[1]
    cross = relative[0] * direction[1] - direction[0] * relative[1]
    cross_sign = jnp.sign(cross)

    dist_p1 = cross_sign * jnp.linalg.norm(relative)
    dist_p2 = cross_sign * jnp.linalg.norm(self.p2 - point)

    return lax.cond(
        projection < 0., true_fn_outer, false_fn_outer, projection, cross, dist_p1, dist_p2
    )

  # [VMAP DOES NOT WORK]
  @partial(jit, static_argnums=(0,))
  def signed_distance_to_vmap(self, points: ArrayImpl) -> ArrayImpl:
    _jitted_fn = jit(jax.vmap(self.signed_distance_to, in_axes=(1), out_axes=(1)))
    return _jitted_fn(points)


# [SLOW JAX COMPUTATION]
class Polyline_jitted(object):
  """
  Polyline class to represent piecewise linear path in 2D.
  """

  def __init__(self, points: ArrayImpl = None):
    """
    Initialization.

    Args:
        points (ArrayImpl, optional): (2, N) Defaults to None.
    """
    self.points = points
    self.N_points = points.shape[1]

  @partial(jit, static_argnums=(0,))
  def signed_distance_to(self, point):
    """
    Compute signed distance from this polyline to the given point.
    Sign convention is positive to the right and negative to the left, e.g.:
                                    *
                                    |
                negative            |             positive
                                    |
                                    |
                                    *

    Args:
        point (ArrayImpl): query point (2,)

    Returns:
        ArrayImpl: scalar
    """

    # NOTE: for now, we'll just implement this with a naive linear search.
    # In future, if we want to optimize at all we can pass in a guess of
    # which index we expect the closest point to be close to.

    def signed_distance_looper(i, abs_signed_distance_array):

      def true_fn_outer(projection, cross, dist_p1, dist_p2):
        return dist_p1

      def false_fn_outer(projection, cross, dist_p1, dist_p2):

        def true_fn_inner(projection, cross, dist_p1, dist_p2):
          return dist_p2

        def false_fn_inner(projection, cross, dist_p1, dist_p2):
          return cross

        return lax.cond(
            projection > length, true_fn_inner, false_fn_inner, projection, cross, dist_p1, dist_p2
        )

      p1 = self.points[:, i]
      p2 = self.points[:, i + 1]
      length = jnp.linalg.norm(p1 - p2)

      # Vector from p1 to query.
      relative = point - p1

      # Compute the unit direction of this line segment.
      direction = p2 - p1
      direction /= jnp.linalg.norm(direction)

      # Find signed length of projection and of cross product.
      projection = relative[0] * direction[0] + relative[1] * direction[1]
      cross = relative[0] * direction[1] - direction[0] * relative[1]
      cross_sign = jnp.sign(cross)

      dist_p1 = cross_sign * jnp.linalg.norm(relative)
      dist_p2 = cross_sign * jnp.linalg.norm(p2 - point)

      abs_signed_distance = jnp.abs(
          lax.cond(
              projection < 0., true_fn_outer, false_fn_outer, projection, cross, dist_p1, dist_p2
          )
      )

      abs_signed_distance_array = abs_signed_distance_array.at[i].set(abs_signed_distance)

      return abs_signed_distance_array

    abs_signed_distance_array = jnp.zeros(self.N_points - 1,)
    abs_signed_distance_array = lax.fori_loop(
        0, self.N_points - 1, signed_distance_looper, abs_signed_distance_array
    )

    return jnp.min(abs_signed_distance_array)
