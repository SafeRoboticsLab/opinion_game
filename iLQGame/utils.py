"""
Util functions for iLQGame.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
Reference: ilqgames/python (David Fridovich-Keil, Ellis Ratner)

TODO:
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


def wrapPi(angle):
  """
  Makes a number -pi to pi.
  """
  while angle <= -np.pi:
    angle += 2 * np.pi
  while angle > np.pi:
    angle -= 2 * np.pi
  return angle
