from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Configuration class."""

import bisect
from collections import deque
import pickle
import heapq
import random
import ast

import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

import logging
logger = logging.getLogger(f'cibi.{__file__}')

from importlib_metadata import version  

import cibi

def trusted_version(experiment_summary):
  """Check if we need to distrust an artifact because it was produced by a previous major version of cibi"""

  if 'cibi-version' not in experiment_summary:
    return False

  artefact_version = [int(x) for x in experiment_summary['cibi-version'].split('.')]
  cibi_version = [int(x) for x in version('cibi').split('.')]
  
  return artefact_version[0] == cibi_version[0]

def with_graph(graph):
  def with_this_graph(method):
    @functools.wraps(method)
    def execute_with_graph():
      with graph.as_default():
        return method()
    return execute_with_graph
  return with_this_graph

def tuple_to_record(tuple_, record_type):
  return record_type(**dict(zip(record_type.__slots__, tuple_)))


def make_record(type_name, attributes, defaults=None):
  """Factory for mutable record classes.

  A record acts just like a collections.namedtuple except slots are writable.
  One exception is that record classes are not equivalent to tuples or other
  record classes of the same length.

  Note, each call to `make_record` produces a unique type. Two calls will make
  different types even if `type_name` is the same each time.

  Args:
    type_name: Name of the record type to create.
    attributes: List of names of each record attribute. The order of the list
        is preserved.
    defaults: (optional) default values for attributes. A dict mapping attribute
        names to values.

  Returns:
    A new record type.

  Raises:
    ValueError: If,
        `defaults` is not a dict,
        `attributes` contains duplicate names,
        `defaults` keys are not contained in `attributes`.
  """
  if defaults is None:
    defaults = {}
  if not isinstance(defaults, dict):
    raise ValueError('defaults must be a dict.')
  attr_set = set(attributes)
  if len(attr_set) < len(attributes):
    raise ValueError('No duplicate attributes allowed.')
  if not set(defaults.keys()).issubset(attr_set):
    raise ValueError('Default attributes must be given in the attributes list.')

  class RecordClass(object):
    """A record type.

    Acts like mutable tuple with named slots.
    """
    __slots__ = list(attributes)
    _defaults = dict(defaults)

    def __init__(self, *args, **kwargs):
      if len(args) > len(self.__slots__):
        raise ValueError('Too many arguments. %s has length %d.'
                         % (type(self).__name__, len(self.__slots__)))
      for attr, val in self._defaults.items():
        setattr(self, attr, val)
      for i, arg in enumerate(args):
        setattr(self, self.__slots__[i], arg)
      for attr, val in kwargs.items():
        setattr(self, attr, val)
      for attr in self.__slots__:
        if not hasattr(self, attr):
          raise ValueError('Required attr "%s" is not set.' % attr)

    def __len__(self):
      return len(self.__slots__)

    def __iter__(self):
      for attr in self.__slots__:
        yield getattr(self, attr)

    def __getitem__(self, index):
      return getattr(self, self.__slots__[index])

    def __setitem__(self, index, value):
      return setattr(self, self.__slots__[index], value)

    def __eq__(self, other):
      # Types must be equal as well as values.
      return (isinstance(other, type(self))
              and all(a == b for a, b in zip(self, other)))

    def __str__(self):
      return '%s(%s)' % (
          type(self).__name__,
          ', '.join(attr + '=' + str(getattr(self, attr))
                    for attr in self.__slots__))

    def __repr__(self):
      return str(self)

  RecordClass.__name__ = type_name
  return RecordClass


# Making minibatches.
def stack_pad(tensors, pad_axes=None, pad_to_lengths=None, dtype=np.float32,
              pad_value=0):
  """Stack tensors along 0-th dim and pad them to be the same shape.

  Args:
    tensors: Any list of iterables (python list, numpy array, etc). Can be 1D
        or multi-D iterables.
    pad_axes: An int or list of ints. Axes to pad along.
    pad_to_lengths: Length in each dimension. If pad_axes was an int, this is an
        int or None. If pad_axes was a list of ints, this is a list of mixed int
        and None types with the same length, or None. A None length means the
        maximum length among the given tensors is used.
    dtype: Type of output numpy array. Defaults to np.float32.
    pad_value: Value to use for padding. Defaults to 0.

  Returns:
    Numpy array containing the tensors stacked along the 0-th dimension and
        padded along the specified dimensions.

  Raises:
    ValueError: If the tensors do not have equal shapes along non-padded
        dimensions.
  """
  tensors = [np.asarray(t) for t in tensors]
  max_lengths = [max(l) for l in zip(*[t.shape for t in tensors])]
  same_axes = dict(enumerate(max_lengths))
  if pad_axes is None:
    pad_axes = []
  if isinstance(pad_axes, six.integer_types):
    if pad_to_lengths is not None:
      max_lengths[pad_axes] = pad_to_lengths
    del same_axes[pad_axes]
  else:
    if pad_to_lengths is None:
      pad_to_lengths = [None] * len(pad_axes)
    for i, l in zip(pad_axes, pad_to_lengths):
      if l is not None:
        max_lengths[i] = l
      del same_axes[i]
  same_axes_items = same_axes.items()
  dest = np.full([len(tensors)] + max_lengths, pad_value, dtype=dtype)
  for i, t in enumerate(tensors):
    for j, l in same_axes_items:
      if t.shape[j] != l:
        raise ValueError(
            'Tensor at index %d does not have size %d along axis %d'
            % (i, l, j))
    dest[[i] + [slice(0, d) for d in t.shape]] = t
  return dest


def first(coll):
  """Get the first element of anything"""
  # It is really frustrating that Python doesn't have this built in

  try:
    return coll[0]
  except TypeError:
    return next(iter(coll))

def alternative_names(name):
    alternative = name
    while True:
        alternative = '_' + alternative
        yield alternative

def get_dir_out_of_the_way(path):
    import os
    import shutil

    if not os.path.exists(path):
      return

    parent, child = os.path.split(path)
    for alternative_name in alternative_names(child):
        alternative_path = os.path.join(parent, alternative_name)
        if not os.path.exists(alternative_path):
            shutil.move(path, alternative_path)
            break

def parse_config_string(config_str):
  config = {}
  for config_statement in config_str.split(','):
    if config_statement:
      key, value = config_statement.split('=')
      config[key] = ast.literal_eval(value)
  return config

def ensure_enough_test_runs(codebase, env, observation_discretizer, action_sampler, runs=100, render=False):
  from cibi import bf
  assert codebase.deduplication

  for code, count, result in zip(codebase['code'], codebase['count'], codebase['result']):
    try:
      if result in ['syntax-error', 'step-limit']:
        continue

      program = bf.Executable(code, observation_discretizer, action_sampler, cycle=True, debug=False)
      for _ in range(runs - count):
        rollout = program.attend_gym(env, render=render)
        codebase.commit(code, metrics={'total_reward': rollout.total_reward})
    except KeyboardInterrupt:
      logger.info('Testing phase cut short by KeyboardInterrupt')
      break

def calc_hash(val):
  import hashlib
  return hashlib.sha224(str(val).encode('utf-8')).hexdigest()

def retry(f, test=lambda x: True, attempts=3, exceptions=BaseException):
  import traceback

  def f_with_retries(*args, **kwargs):
    result = None

    for idx in range(attempts - 1):
      try:
        result = f(*args, **kwargs)
        if test(result):
          return result
      except exceptions:
        logger.error(traceback.format_exc())
        pass

    return f(*args, **kwargs)
  return f_with_retries

def update_keys(dictionary, f):
    if type(dictionary) is dict:
        return {f(key): update_keys(val, f) for key, val in dictionary.items()}
    else:
        return dictionary