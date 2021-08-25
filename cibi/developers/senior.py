import tensorflow as tf
from tensorflow.python.client import device_lib

import logging
import itertools
import pickle
import os
import time
import numpy as np
from collections import namedtuple

from cibi import utils

logger = logging.getLogger(f'cibi.{__file__}')

Reinforcement = namedtuple(
    'Reinforcement',
    ['episode_count', 'episode_lengths', 'episode_code_strings', 
     'episode_actions', 'action_rewards', 'action_log_probs', 
     'episode_rewards', 'episode_values', 'episode_results'])

def make_initialized_variable(value, name, shape=None, dtype=tf.float32):
  """Create a tf.Variable with a constant initializer.

  Args:
    value: Constant value to initialize the variable with. This is the value
        that the variable starts with.
    name: Name of the variable in the TF graph.
    shape: Shape of the variable. If None, variable will be a scalar.
    dtype: Data type of the variable. Should be a TF dtype. Defaults to
        tf.float32.

  Returns:
    tf.Variable instance.
  """
  if shape is None:
    shape = []
  return tf.get_variable(
      name=name, shape=shape, initializer=tf.constant_initializer(value),
      dtype=dtype, trainable=False)

# TODO: see if we can use XLA
devices = [device for device in device_lib.list_local_devices() 
                  if device.device_type[:3] != 'XLA']
gpus_available = any(device.device_type != 'CPU' for device in devices)
if gpus_available:
  # Only use GPUs
  devices = [device for device in devices if device.device_type != 'CPU']
device_cycle = itertools.cycle(devices)

class SeniorDeveloper(object):
  """Writes code using 2 language models

  A global model on the parameter server, and a local
  model (for this worker). Gradient updates are sent to the global model, and
  the updated weights are synced to the local copy.
  """

  def __init__(self, config,
               make_language_model,
               name='senior'):
    self.name = name 
    self._make_language_model = make_language_model
    self.config = config

  def set_language(self, language, 
                   summary_writer=None, dtype=tf.float32,
                   summary_interval=1, run_number=0, model_v=0):
    self.graph = tf.Graph()
    
    with self.graph.as_default():

      worker_device = next(device_cycle)
      logger.info('worker_device: %s', worker_device.name)
      tf.get_variable_scope().set_use_resource(True)

      # global model
      with tf.device(worker_device.name):
        with tf.variable_scope('global'):
          global_model = self._make_language_model(language, self.config, dtype=dtype, is_local=False)
          global_params_dict = {p.name: p
                                for p in global_model.sync_variables}
          self.global_model = global_model
          self.global_step = make_initialized_variable(
              0, 'global_step', dtype=tf.int64)

          self.global_best_reward = make_initialized_variable(
              -10.0, 'global_best_reward', dtype=tf.float64)
          self.is_best_model = make_initialized_variable(
              False, 'is_best_model', dtype=tf.bool)
          self.reset_is_best_model = self.is_best_model.assign(False)
          self.global_best_reward_placeholder = tf.placeholder(
              tf.float64, [], name='global_best_reward_placeholder')
          self.assign_global_best_reward_op = tf.group(
              self.global_best_reward.assign(
                  self.global_best_reward_placeholder),
              self.is_best_model.assign(True))
          def assign_global_best_reward_fn(session, reward):
            reward = round(reward, 10)
            best_reward = round(session.run(self.global_best_reward), 10)
            is_best = reward > best_reward
            if is_best:
              session.run(self.assign_global_best_reward_op,
                          {self.global_best_reward_placeholder: reward})
            return is_best
          self.assign_global_best_reward_fn = assign_global_best_reward_fn

          self.run_number = make_initialized_variable(
              run_number, 'run_number', dtype=tf.int32)

          # Count all programs sampled from policy. This does not include
          # programs sampled from replay buffer.
          # This equals NPE (number of programs executed). Only programs sampled
          # from the policy need to be executed.
          self.program_count = make_initialized_variable(
              0, 'program_count', dtype=tf.int64)

      # local model
      with tf.device(worker_device.name):
        with tf.variable_scope('local'):
          self.model = model = self._make_language_model(
              language, 
              self.config,
              dtype=dtype,
              global_best_reward_fn=self.assign_global_best_reward_fn,
              program_count=self.program_count,
              verbose_level=model_v)
          local_params = model.trainable_variables
          local_params_dict = {p.name: p for p in local_params}

      # Pull global params to local model.
      def _global_to_local_scope(name):
        assert name.startswith('global/')
        return 'local' + name[6:]
      sync_dict = {
          local_params_dict[_global_to_local_scope(p_name)]: p
          for p_name, p in global_params_dict.items()}
      self.sync_op = tf.group(*[v_local.assign(v_global)
                                for v_local, v_global
                                in sync_dict.items()])

      # Pair local gradients with global params.
      grad_var_dict = {
          gradient: sync_dict[local_var]
          for local_var, gradient in model.gradients_dict.items()}

      # local model
      model.make_summary_ops()  # Don't put summaries under 'local' scope.
      with tf.variable_scope('local'):
        self.train_op = model.optimizer.apply_gradients(
            grad_var_dict.items(), global_step=self.global_step)
        self.local_init_op = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              tf.get_variable_scope().name))

      self.local_step = 0
      self.last_summary_time = time.time()
      self.summary_interval = summary_interval
      self.summary_writer = summary_writer
      self.cached_global_step = -1
      self.cached_global_npe = -1

      logger.info('summary_interval: %d', self.summary_interval)

      variables_to_save = [v for v in tf.global_variables()
                          if v.name.startswith('global')]
      self.ready_op = tf.report_uninitialized_variables(variables_to_save)
      self.global_init_op = tf.variables_initializer(variables_to_save)
      self.saver = tf.train.Saver(variables_to_save)

  def initialize(self, session):
    """Run initialization ops."""
    session.run(self.local_init_op)
    session.run(self.sync_op)
    self.cached_global_step, self.cached_global_npe = session.run(
        [self.global_step, self.program_count])

  def write_programs(self, session, inspiration_branch):
    session.run(self.sync_op)  # Copy weights from global to local.

    with session.as_default():
      return self.model.write_programs(session, inspiration_branch)

  def _log_reflection_result(self, session, reflection_result):
    global_step = reflection_result.global_step
    summaries = reflection_result.summaries_list

    if self.summary_writer and self.local_step % self.summary_interval == 0:
      if not isinstance(summaries, (tuple, list)):
        summaries = [summaries]
      summaries.append(self._local_step_summary())
      (global_best_reward,
        program_count) = session.run(
            [self.global_best_reward,
            self.program_count])
      summaries.append(
          tf.Summary(
              value=[tf.Summary.Value(
                  tag='model/best_reward',
                  simple_value=global_best_reward)]))
      summaries.append(
          tf.Summary(
              value=[tf.Summary.Value(
                  tag='model/program_count',
                  simple_value=program_count)]))
      for s in summaries:
        self.summary_writer.add_summary(s, global_step)
      self.last_summary_time = time.time()

  def accept_feedback(self, session, feedback_branch):
    """Run an update step.

    1) Asynchronously copy global weights to local model.
    2) Call into local model's update_step method, which does the following:
        a) Sample batch of programs from policy.
        b) Compute rewards.
        c) Compute gradients and update the global model asynchronously.
    3) Write tensorboard summaries to disk.

    Args:
      session: tf.Session instance.
    """
    session.run(self.sync_op)  # Copy weights from global to local.

    with session.as_default():
      result = self.model.accept_feedback(
          session, feedback_branch, 
          self.train_op, self.global_step)
      global_step = result.global_step
      global_npe = result.global_npe
    self.cached_global_step = global_step
    self.cached_global_npe = global_npe
    self.local_step += 1

    self._log_reflection_result(session, result)

  def _local_step_summary(self):
    """Compute number of local steps per time increment."""
    dt = time.time() - self.last_summary_time
    steps_per_time = self.summary_interval / float(dt)
    return tf.Summary(value=[
        tf.Summary.Value(
            tag='local_step/per_sec',
            simple_value=steps_per_time),
        tf.Summary.Value(
            tag='local_step/step',
            simple_value=self.local_step)])

  def hire(self, language, log_dir, events_dir=None, is_chief=True):
    self.set_language(language)

    with self.graph.as_default():
      summary_writer = tf.summary.FileWriter(os.path.join(events_dir, self.name)) if events_dir else None

      sv = tf.train.Supervisor( is_chief=is_chief,
                                logdir=os.path.join(log_dir, self.name),
                                saver=self.saver,
                                summary_op=None,
                                init_op=self.global_init_op,
                                init_fn=init_fn,
                                summary_writer=summary_writer,
                                ready_op=self.ready_op,
                                ready_for_local_init_op=None,
                                global_step=self.global_step,
                                save_model_secs=30,
                                save_summaries_secs=30)

      config = tf.ConfigProto(allow_soft_placement=True,
                              gpu_options=tf.GPUOptions(allow_growth=True))
      return EmployedDeveloper(self, sv.managed_session(config=config))

def init_fn(unused_sess):
  logger.info('No checkpoint found. Initialized global params.')

class EmployedDeveloper():
  """
  A senior developer compatible with Scrum Master
  """

  def __init__(self, developer, session_manager):
    self.developer = developer
    self.name = developer.name
    self.session_manager = session_manager

  def write_programs(self, inspiration_branch):
    return self.developer.write_programs(self.session, inspiration_branch)

  def accept_feedback(self, feedback_branch):
    return self.developer.accept_feedback(self.session, feedback_branch)

  def __enter__(self):
    self.session = self.session_manager.__enter__()
    self.developer.initialize(self.session)
    return self 

  def __exit__(self, type, value, tb):
    self.session_manager.__exit__(type, value, tb)
    self.session = None