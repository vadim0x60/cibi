import tensorflow as tf
import logging
import pickle
import os
import time
import numpy as np
from collections import namedtuple

from cibi import utils

logger = logging.getLogger(f'bff.{__file__}')

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

class Developer(object):
  """Writes code using 2 language models

  A global model on the parameter server, and a local
  model (for this worker). Gradient updates are sent to the global model, and
  the updated weights are synced to the local copy.
  """

  def __init__(self, config,
               make_language_model,
               task_id=0, ps_tasks=0, num_workers=1, is_chief=True,
               summary_writer=None,
               dtype=tf.float32,
               summary_interval=1,
               run_number=0,
               logging_dir='/tmp', model_v=0):
    self.batch_size = config.batch_size
    self.task_id = task_id
    self.ps_tasks = ps_tasks
    self.is_chief = is_chief

    if ps_tasks == 0:
      assert task_id == 0, 'No parameter servers specified. Expecting 1 task.'
      assert num_workers == 1, (
          'No parameter servers specified. Expecting 1 task.')
      worker_device = '/job:localhost/replica:%d/task:0/cpu:0' % task_id
      # worker_device = '/cpu:0'
      # ps_device = '/cpu:0'
    else:
      assert num_workers > 0, 'There must be at least 1 training worker.'
      worker_device = '/job:worker/replica:%d/task:0/cpu:0' % task_id
      # ps_device = '/job:ps/replica:0/task:0/cpu:0'
    logger.info('worker_device: %s', worker_device)

    logging_file = os.path.join(
        logging_dir, 'solutions_%d.txt' % task_id)
    experience_replay_file = os.path.join(
        logging_dir, 'replay_buffer_%d.pickle' % task_id)
    self.topk_file = os.path.join(
        logging_dir, 'topk_buffer_%d.pickle' % task_id)

    tf.get_variable_scope().set_use_resource(True)

    # global model
    with tf.device(tf.train.replica_device_setter(ps_tasks,
                                                  ps_device='/job:ps/replica:0',
                                                  worker_device=worker_device)):
      with tf.variable_scope('global'):
        global_model = make_language_model(config, dtype=dtype, is_local=False)
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
    with tf.device(worker_device):
      with tf.variable_scope('local'):
        self.model = model = make_language_model(
            config,
            logging_file=logging_file,
            experience_replay_file=experience_replay_file,
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

    # Load top-k buffer.
    if self.model.top_episodes is not None and tf.gfile.Exists(self.topk_file):
      try:
        with tf.gfile.FastGFile(self.topk_file, 'r') as f:
          self.model.top_episodes = pickle.loads(f.read())
        logger.info(
            'Loaded top-k buffer from disk with %d items. Location: "%s"',
            len(self.model.top_episodes), self.topk_file)
      except (pickle.UnpicklingError, EOFError) as e:
        logger.warn(
            'Failed to load existing top-k buffer from disk. Removing bad file.'
            '\nLocation: "%s"\nException: %s', self.topk_file, str(e))
        tf.gfile.Remove(self.topk_file)

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

  def write_programs(self, session):
    session.run(self.sync_op)  # Copy weights from global to local.

    with session.as_default():
      return self.model.write_programs(session)

  def _log_reflection_result(self, session, reflection_result):
    global_step = reflection_result.global_step
    summaries = reflection_result.summaries_list

    if self.summary_writer and self.local_step % self.summary_interval == 0:
      if not isinstance(summaries, (tuple, list)):
        summaries = [summaries]
      summaries.append(self._local_step_summary())
      if self.is_chief:
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

  def reflect(self, session, reinforcement):
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
      result = self.model.reflect(session, reinforcement, self.train_op,
          self.global_step)
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

  def save_replay_buffer(self):
    """Save replay buffer to disk.

    Call this periodically so that training can recover if jobs go down.
    """
    if self.model.experience_replay is not None:
      logger.info('Saving experience replay buffer to "%s".',
                   self.model.experience_replay.save_file)
      self.model.experience_replay.incremental_save(True)

  def delete_replay_buffer(self):
    """Delete replay buffer from disk.

    Call this at the end of training to clean up. Replay buffer can get very
    large.
    """
    if self.model.experience_replay is not None:
      logger.info('Deleting experience replay buffer at "%s".',
                   self.model.experience_replay.save_file)
      tf.gfile.Remove(self.model.experience_replay.save_file)

  def save_topk_buffer(self):
    """Save top-k buffer to disk.

    Call this periodically so that training can recover if jobs go down.
    """
    if self.model.top_episodes is not None:
      logger.info('Saving top-k buffer to "%s".', self.topk_file)
      # Overwrite previous data each time.
      with tf.gfile.FastGFile(self.topk_file, 'w') as f:
        f.write(pickle.dumps(self.model.top_episodes))

def init_fn(unused_sess):
  logger.info('No checkpoint found. Initialized global params.')

class EmployedDeveloper():
  """
  A developer with an active Tensorflow session
  """

  def __init__(self, developer, session_manager):
    self.developer = developer
    self.session_manager = session_manager
    self.batch_size = developer.batch_size

  def write_programs(self):
    return self.developer.write_programs(self.session)

  def reflect(self, programs):
    return self.developer.reflect(self.session, programs)

  def __enter__(self):
    self.session = self.session_manager.__enter__()
    self.developer.initialize(self.session)
    return self 

  def __exit__(self, type, value, tb):
    self.session_manager.__exit__(type, value, tb)
    self.session = None

def hire(developer, log_dir, events_dir=None, is_chief=True):
  summary_writer = tf.summary.FileWriter(events_dir) if events_dir else None

  sv = tf.train.Supervisor( is_chief=is_chief,
                            logdir=log_dir,
                            saver=developer.saver,
                            summary_op=None,
                            init_op=developer.global_init_op,
                            init_fn=init_fn,
                            summary_writer=summary_writer,
                            ready_op=developer.ready_op,
                            ready_for_local_init_op=None,
                            global_step=developer.global_step,
                            save_model_secs=30,
                            save_summaries_secs=30)

  return EmployedDeveloper(developer, sv.managed_session())