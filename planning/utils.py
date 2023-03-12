"""
Util functions for planning.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

from functools import partial
from jax import jit
from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp


# @partial(jit, static_argnums=(1,))
# def softmax(z: DeviceArray, idx: int) -> DeviceArray:
#   """
#   Softmax operator.

#   Args:
#       z (DeviceArray): vector.
#       idx (int): index.

#   Returns:
#       DeviceArray: output.
#   """
#   return jnp.exp(z[idx]) / sum(jnp.exp(z))
