"""
Util functions for opinion dynamics.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

from functools import partial
from jax import jit
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp


@partial(jit, static_argnums=(1,))
def softmax(z: ArrayImpl, idx: int) -> ArrayImpl:
  """
  Softmax operator.

  Args:
      z (ArrayImpl): vector
      idx (int): index

  Returns:
      ArrayImpl: output
  """
  return jnp.exp(z[idx]) / jnp.sum(jnp.exp(z))
