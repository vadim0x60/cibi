from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Language model agent.

Agent outputs code in a sequence just like a language model. Can be trained
as a language model or using RL, or a combination of the two.
"""

from collections import namedtuple
from math import exp, log
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import cibi.rollout as rollout_lib  # brain coder
from cibi import utils
from cibi import bf
from cibi.codebase import make_dev_codebase, make_codebase_like, Codebase

import logging
logger = logging.getLogger(f'cibi.{__file__}')

# Experiments in the ICLR 2018 paper used reduce_sum instead of reduce_mean for
# some losses. We make all loses be batch_size independent, and multiply the
# changed losses by 64, which was the fixed batch_size when the experiments
# where run. The loss hyperparameters still match what is reported in the paper.
MAGIC_LOSS_MULTIPLIER = 64

def rshift_time(tensor_2d, fill):
  """Right shifts a 2D tensor along the time dimension (axis-1)."""
  dim_0 = tf.shape(tensor_2d)[0]
  fill_tensor = tf.fill([dim_0, 1], fill)
  return tf.concat([fill_tensor, tensor_2d[:, :-1]], axis=1)

ReflectionResult = namedtuple(
    'ReflectionResult',
    ['global_step', 'global_npe', 'summaries_list', 'gradients_dict'])

LMConfig = namedtuple(
  'NeuralParams',
  ['policy_lstm_sizes', 'obs_embedding_size', 'grad_clip_threshold', 
   'param_init_factor', 'lr', 'pi_loss_hparam', 'entropy_beta', 'regularizer',
   'softmax_tr', 'optimizer', 'topk', 'topk_loss_hparam', 'topk_batch_size',
   'ema_baseline_decay', 'alpha', 'iw_normalize', 'batch_size', 'timestep_limit']
)

default_config = {
  'batch_size': 4,
  'policy_lstm_sizes': [50,50],
  'obs_embedding_size': 15,
  'grad_clip_threshold': 10.0,
  'param_init_factor':1.0,
  'lr':5e-5,
  'pi_loss_hparam':1.0,
  'entropy_beta':1e-2,
  'regularizer':0.0,
  'softmax_tr':1.0,  # Reciprocal temperature.
  'optimizer':'rmsprop',  # 'adam', 'sgd', 'rmsprop'
  'topk':5,  # Top-k unique codes will be stored.
  'topk_loss_hparam':0.5,  # off policy loss multiplier.
  # Uniformly sample this many episodes from topk buffer per batch.
  # If topk is 0, this has no effect.
  'topk_batch_size':1,
  'timestep_limit': 1024,
  # Exponential moving average baseline for REINFORCE.
  # If zero, A2C is used.
  # If non-zero, should be close to 1, like .99, .999, etc.
  'ema_baseline_decay':0.99,
  # Replay probability. 1 : always replay, 0 : always on policy.
  'alpha':0.75,
  # Whether to normalize importance weights in each minibatch.
  'iw_normalize':True
}

def list_(x):
  return [] if x is None else list(x)

def make_optimizer(kind, lr):
  if kind == 'sgd':
    return tf.train.GradientDescentOptimizer(lr)
  elif kind == 'adam':
    return tf.train.AdamOptimizer(lr)
  elif kind == 'rmsprop':
    return tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99)
  else:
    raise ValueError('Optimizer type "%s" not recognized.' % kind)

class LinearWrapper(tf.contrib.rnn.RNNCell):
  """RNNCell wrapper that adds a linear layer to the output."""

  def __init__(self, cell, output_size, dtype=tf.float32, suppress_index=None):
    self.cell = cell
    self._output_size = output_size
    self._dtype = dtype
    self._suppress_index = suppress_index
    self.smallest_float = -2.4e38

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(type(self).__name__):
      outputs, state = self.cell(inputs, state, scope=scope)
      logits = tf.matmul(
          outputs,
          tf.get_variable('w_output',
                          [self.cell.output_size, self.output_size],
                          dtype=self._dtype))
      if self._suppress_index is not None:
        # Replace the target index with -inf, so that it never gets selected.
        batch_size = tf.shape(logits)[0]
        logits = tf.concat(
            [logits[:, :self._suppress_index],
             tf.fill([batch_size, 1], self.smallest_float),
             logits[:, self._suppress_index + 1:]],
            axis=1)

    return logits, state

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self.cell.state_size

  def zero_state(self, batch_size, dtype):
    return self.cell.zero_state(batch_size, dtype)


class AttrDict(dict):
  """Dict with attributes as keys.

  https://stackoverflow.com/a/14620633
  """

  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


class LanguageModel:
  """Language model agent."""
  def __init__(self, language, config={},
               global_best_reward_fn=None,
               program_count=None,
               do_iw_summaries=False,
               dtype=tf.float32,
               verbose_level=0,
               is_local=True):
    if type(config) == LMConfig:
      self.config = config
    elif type(config) == dict:
      self.config = LMConfig(**{**default_config, **config})
      config = None # prevent accidentaly using partial config
    else:
      raise ValueError('Unsupported configuration format. Should be a dict or LMConfigs')

    self.verbose_level = verbose_level
    self.global_best_reward_fn = global_best_reward_fn
    self.parent_scope_name = tf.get_variable_scope().name
    self.dtype = dtype
    self.is_local = is_local

    self.top_reward = 0.0
    self.embeddings_trainable = True

    self.action_space = len(language['alphabet'])
    self.observation_space = len(language['alphabet'])
    self.int_to_char = language['int_to_char']
    self.char_to_int = language['char_to_int']
    self.eos_char = language['eos_char']
    self.eos_int = language['eos_int']    

    self.no_op = tf.no_op()

    self.learning_rate = tf.constant(
        self.config.lr, dtype=dtype, name='learning_rate')
    self.initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=self.config.param_init_factor,
        mode='FAN_AVG',
        uniform=True,
        dtype=dtype)  # TF's default initializer.
    tf.get_variable_scope().set_initializer(self.initializer)

    logger.info('Using exponential moving average REINFORCE baselines.')
    self.ema_by_len = [0.0] * self.config.timestep_limit

    # Top-k
    topk_params = (self.config.topk, self.config.topk_loss_hparam, self.config.topk_batch_size)
    if any(p != 0 for p in topk_params):
      assert all(p > 0 for p in topk_params)

    # Experience replay.
    if self.config.alpha == 0:
      self.num_replay_per_batch = 0
    else:
      self.num_replay_per_batch = int(self.config.batch_size * self.config.alpha)
    self.num_on_policy_per_batch = (
        self.config.batch_size - self.num_replay_per_batch)
    self.replay_alpha = (
        self.num_replay_per_batch / float(self.config.batch_size))
    logger.info('num_replay_per_batch: %d', self.num_replay_per_batch)
    logger.info('num_on_policy_per_batch: %d', self.num_on_policy_per_batch)
    logger.info('replay_alpha: %s', self.replay_alpha)

    if program_count is not None:
      self.program_count = program_count
      self.program_count_add_ph = tf.placeholder(
          tf.int64, [], 'program_count_add_ph')
      self.program_count_add_op = self.program_count.assign_add(
          self.program_count_add_ph)

    ################################
    # RL policy network #
    ################################
    batch_size = self.config.batch_size
    logger.info('batch_size: %d', batch_size)

    self.policy_cell = LinearWrapper(
        tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(cell_size)
             for cell_size in self.config.policy_lstm_sizes]),
        self.action_space,
        dtype=dtype,
        suppress_index=None)

    obs_embedding_scope = 'obs_embed'
    with tf.variable_scope(
        obs_embedding_scope,
        initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)):
      obs_embeddings = tf.get_variable(
          'embeddings',
          [self.observation_space, self.config.obs_embedding_size],
          dtype=dtype, trainable=self.embeddings_trainable)
      self.obs_embeddings = obs_embeddings

    ################################
    # RL policy and value networks #
    ################################

    initial_state = tf.fill([batch_size], self.eos_int)
    def loop_fn(loop_time, cell_output, cell_state, loop_state):
      """Function called by tf.nn.raw_rnn to instantiate body of the while_loop.

      See https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn for more
      information.

      When time is 0, and cell_output, cell_state, loop_state are all None,
      `loop_fn` will create the initial input, internal cell state, and loop
      state. When time > 0, `loop_fn` will operate on previous cell output,
      state, and loop state.

      Args:
        loop_time: A scalar tensor holding the current timestep (zero based
            counting).
        cell_output: Output of the raw_rnn cell at the current timestep.
        cell_state: Cell internal state at the current timestep.
        loop_state: Additional loop state. These tensors were returned by the
            previous call to `loop_fn`.

      Returns:
        elements_finished: Bool tensor of shape [batch_size] which marks each
            sequence in the batch as being finished or not finished.
        next_input: A tensor containing input to be fed into the cell at the
            next timestep.
        next_cell_state: Cell internal state to be fed into the cell at the
            next timestep.
        emit_output: Tensor to be added to the TensorArray returned by raw_rnn
            as output from the while_loop.
        next_loop_state: Additional loop state. These tensors will be fed back
            into the next call to `loop_fn` as `loop_state`.
      """
      if cell_output is None:  # 0th time step.
        next_cell_state = self.policy_cell.zero_state(batch_size, dtype)
        elements_finished = tf.zeros([batch_size], tf.bool)
        output_lengths = tf.ones([batch_size], dtype=tf.int32)
        next_input = tf.gather(obs_embeddings, initial_state)
        emit_output = None
        next_loop_state = (
            tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True),
            output_lengths,
            elements_finished
        )
      else:
        scaled_logits = cell_output * self.config.softmax_tr  # Scale temperature.
        prev_chosen, prev_output_lengths, prev_elements_finished = loop_state
        next_cell_state = cell_state
        chosen_outputs = tf.to_int32(tf.where(
            tf.logical_not(prev_elements_finished),
            tf.multinomial(logits=scaled_logits, num_samples=1)[:, 0],
            tf.zeros([batch_size], dtype=tf.int64)))
        elements_finished = tf.logical_or(
            tf.equal(chosen_outputs, self.eos_int),
            loop_time >= self.config.timestep_limit)
        output_lengths = tf.where(
            elements_finished,
            prev_output_lengths,
            # length includes EOS token. empty seq has len 1.
            tf.tile(tf.expand_dims(loop_time + 1, 0), [batch_size])
        )
        next_input = tf.gather(obs_embeddings, chosen_outputs)
        emit_output = scaled_logits
        next_loop_state = (prev_chosen.write(loop_time - 1, chosen_outputs),
                           output_lengths,
                           tf.logical_or(prev_elements_finished,
                                         elements_finished))
      return (elements_finished, next_input, next_cell_state, emit_output,
              next_loop_state)

    with tf.variable_scope('policy'):
      (decoder_outputs_ta,
       _,  # decoder_state
       (sampled_output_ta, output_lengths, _)) = tf.nn.raw_rnn(
           cell=self.policy_cell,
           loop_fn=loop_fn)
    policy_logits = tf.transpose(decoder_outputs_ta.stack(), (1, 0, 2),
                                 name='policy_logits')
    sampled_tokens = tf.transpose(sampled_output_ta.stack(), (1, 0),
                                  name='sampled_tokens')
    # for sampling actions from the agent, and which told tensors for doing
    # gradient updates on the agent.
    self.sampled_batch = AttrDict(
        logits=policy_logits,
        tokens=sampled_tokens,
        episode_lengths=output_lengths,
        probs=tf.nn.softmax(policy_logits),
        log_probs=tf.nn.log_softmax(policy_logits))

    # adjusted_lengths can be less than the full length of each episode.
    # Use this to train on only part of an episode (starting from t=0).
    self.adjusted_lengths = tf.placeholder(
        tf.int32, [None], name='adjusted_lengths')
    self.policy_multipliers = tf.placeholder(
        dtype,
        [None, None],
        name='policy_multipliers')
    # Empirical value, i.e. discounted sum of observed future rewards from each
    # time step in the episode.
    self.empirical_values = tf.placeholder(
        dtype,
        [None, None],
        name='empirical_values')

    # Off-policy training. Just add supervised loss to the RL loss.
    self.off_policy_targets = tf.placeholder(
        tf.int32,
        [None, None],
        name='off_policy_targets')
    self.off_policy_target_lengths = tf.placeholder(
        tf.int32, [None], name='off_policy_target_lengths')

    self.actions = tf.placeholder(tf.int32, [None, None], name='actions')
    # Add SOS to beginning of the sequence.
    inputs = rshift_time(self.actions, fill=self.eos_int)
    with tf.variable_scope('policy', reuse=True):
      logits, _ = tf.nn.dynamic_rnn(
          self.policy_cell, tf.gather(obs_embeddings, inputs),
          sequence_length=self.adjusted_lengths,
          dtype=dtype)

    self.given_batch = AttrDict(
        logits=logits,
        tokens=sampled_tokens,
        episode_lengths=self.adjusted_lengths,
        probs=tf.nn.softmax(logits),
        log_probs=tf.nn.log_softmax(logits))

    # Episode masks.
    max_episode_length = tf.shape(self.actions)[1]
    # range_row shape: [1, max_episode_length]
    range_row = tf.expand_dims(tf.range(max_episode_length), 0)
    episode_masks = tf.cast(
        tf.less(range_row, tf.expand_dims(self.given_batch.episode_lengths, 1)),
        dtype=dtype)
    episode_masks_3d = tf.expand_dims(episode_masks, 2)

    # Length adjusted episodes.
    self.a_probs = a_probs = self.given_batch.probs * episode_masks_3d
    self.a_log_probs = a_log_probs = (
        self.given_batch.log_probs * episode_masks_3d)
    self.a_policy_multipliers = a_policy_multipliers = (
        self.policy_multipliers * episode_masks)

    # pi_loss is scalar
    acs_onehot = tf.one_hot(self.actions, self.action_space, dtype=dtype)
    self.acs_onehot = acs_onehot
    chosen_masked_log_probs = acs_onehot * a_log_probs
    pi_target = tf.expand_dims(a_policy_multipliers, -1)
    pi_loss_per_step = chosen_masked_log_probs * pi_target  # Maximize.
    self.pi_loss = pi_loss = (
        -tf.reduce_mean(tf.reduce_sum(pi_loss_per_step, axis=[1, 2]), axis=0)
        * MAGIC_LOSS_MULTIPLIER)  # Minimize.
    assert len(self.pi_loss.shape) == 0  # pylint: disable=g-explicit-length-test

    # shape: [batch_size, time]
    self.chosen_log_probs = tf.reduce_sum(chosen_masked_log_probs, axis=2)
    self.chosen_probs = tf.reduce_sum(acs_onehot * a_probs, axis=2)

    # Maximize entropy regularizer
    self.entropy = entropy = (
        -tf.reduce_mean(
            tf.reduce_sum(a_probs * a_log_probs, axis=[1, 2]), axis=0)
        * MAGIC_LOSS_MULTIPLIER)  # Maximize
    self.negentropy = -entropy  # Minimize negentropy.
    assert len(self.negentropy.shape) == 0  # pylint: disable=g-explicit-length-test

    # off-policy loss
    self.offp_switch = tf.placeholder(dtype, [], name='offp_switch')
    if self.config.topk != 0:
      # Add SOS to beginning of the sequence.
      offp_inputs = tf.gather(obs_embeddings,
                              rshift_time(self.off_policy_targets,
                                          fill=self.eos_int))
      with tf.variable_scope('policy', reuse=True):
        offp_logits, _ = tf.nn.dynamic_rnn(
            self.policy_cell, offp_inputs, self.off_policy_target_lengths,
            dtype=dtype)  # shape: [batch_size, time, action_space]
      topk_loss_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.off_policy_targets,
          logits=offp_logits,
          name='topk_loss_per_logit')
      # Take mean over batch dimension so that the loss multiplier strength is
      # independent of batch size. Sum over time dimension.
      topk_loss = tf.reduce_mean(
          tf.reduce_sum(topk_loss_per_step, axis=1), axis=0)
      assert len(topk_loss.shape) == 0  # pylint: disable=g-explicit-length-test
      self.topk_loss = topk_loss * self.offp_switch
      logger.info('Including off policy loss.')
    else:
      self.topk_loss = topk_loss = 0.0

    self.entropy_hparam = tf.constant(
        self.config.entropy_beta, dtype=dtype, name='entropy_beta')

    self.pi_loss_term = pi_loss * self.config.pi_loss_hparam
    self.entropy_loss_term = self.negentropy * self.entropy_hparam
    self.topk_loss_term = self.config.topk_loss_hparam * topk_loss
    self.loss = (
        self.pi_loss_term
        + self.entropy_loss_term
        + self.topk_loss_term)

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               tf.get_variable_scope().name)
    self.trainable_variables = params
    self.sync_variables = self.trainable_variables
    non_embedding_params = [p for p in params
                            if obs_embedding_scope not in p.name]
    self.non_embedding_params = non_embedding_params
    self.params = params

    if self.config.regularizer:
      logger.info('Adding L2 regularizer with scale %.2f.',
                   self.config.regularizer)
      self.regularizer = self.config.regularizer * sum(
          tf.nn.l2_loss(w) for w in non_embedding_params)
      self.loss += self.regularizer
    else:
      logger.info('Skipping regularizer.')
      self.regularizer = 0.0

    # Only build gradients graph for local model.
    if self.is_local:
      unclipped_grads = tf.gradients(self.loss, params)
      self.dense_unclipped_grads = [
          tf.convert_to_tensor(g) for g in unclipped_grads]
      self.grads, self.global_grad_norm = tf.clip_by_global_norm(
          unclipped_grads, self.config.grad_clip_threshold)
      self.gradients_dict = dict(zip(params, self.grads))
      self.optimizer = make_optimizer(self.config.optimizer, self.learning_rate)
      self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             tf.get_variable_scope().name)

    self.do_iw_summaries = do_iw_summaries
    if self.do_iw_summaries:
      b = None
      self.log_iw_replay_ph = tf.placeholder(tf.float32, [b],
                                             'log_iw_replay_ph')
      self.log_iw_policy_ph = tf.placeholder(tf.float32, [b],
                                             'log_iw_policy_ph')
      self.log_prob_replay_ph = tf.placeholder(tf.float32, [b],
                                               'log_prob_replay_ph')
      self.log_prob_policy_ph = tf.placeholder(tf.float32, [b],
                                               'log_prob_policy_ph')
      self.log_norm_replay_weights_ph = tf.placeholder(
          tf.float32, [b], 'log_norm_replay_weights_ph')
      self.iw_summary_op = tf.summary.merge([
          tf.summary.histogram('is/log_iw_replay', self.log_iw_replay_ph),
          tf.summary.histogram('is/log_iw_policy', self.log_iw_policy_ph),
          tf.summary.histogram('is/log_prob_replay', self.log_prob_replay_ph),
          tf.summary.histogram('is/log_prob_policy', self.log_prob_policy_ph),
          tf.summary.histogram(
              'is/log_norm_replay_weights', self.log_norm_replay_weights_ph),
      ])

  def _iw_summary(self, session, replay_iw, replay_log_probs,
                  norm_replay_weights, on_policy_iw,
                  on_policy_log_probs):
    """Compute summaries for importance weights at a given batch.

    Args:
      session: tf.Session instance.
      replay_iw: Importance weights for episodes from replay buffer.
      replay_log_probs: Total log probabilities of the replay episodes under the
          current policy.
      norm_replay_weights: Normalized replay weights, i.e. values in `replay_iw`
          divided by the total weight in the entire replay buffer. Note, this is
          also the probability of selecting each episode from the replay buffer
          (in a roulette wheel replay buffer).
      on_policy_iw: Importance weights for episodes sampled from the current
          policy.
      on_policy_log_probs: Total log probabilities of the on-policy episodes
          under the current policy.

    Returns:
      Serialized TF summaries. Use a summary writer to write these summaries to
      disk.
    """
    return session.run(
        self.iw_summary_op,
        {self.log_iw_replay_ph: np.log(replay_iw),
         self.log_iw_policy_ph: np.log(on_policy_iw),
         self.log_norm_replay_weights_ph: np.log(norm_replay_weights),
         self.log_prob_replay_ph: replay_log_probs,
         self.log_prob_policy_ph: on_policy_log_probs})

  def make_summary_ops(self):
    """Construct summary ops for the model."""
    # size = number of timesteps across entire batch. Number normalized by size
    # will not be affected by the amount of padding at the ends of sequences
    # in the batch.
    size = tf.cast(
        tf.reduce_sum(self.given_batch.episode_lengths), dtype=self.dtype)
    offp_size = tf.cast(tf.reduce_sum(self.off_policy_target_lengths),
                        dtype=self.dtype)
    scope_prefix = self.parent_scope_name

    def _remove_prefix(prefix, name):
      assert name.startswith(prefix)
      return name[len(prefix):]

    # RL summaries.
    self.rl_summary_op = tf.summary.merge(
        [tf.summary.scalar('model/policy_loss', self.pi_loss / size),
         tf.summary.scalar('model/topk_loss', self.topk_loss / offp_size),
         tf.summary.scalar('model/entropy', self.entropy / size),
         tf.summary.scalar('model/loss', self.loss / size),
         tf.summary.scalar('model/grad_norm',
                           tf.global_norm(self.grads)),
         tf.summary.scalar('model/unclipped_grad_norm', self.global_grad_norm),
         tf.summary.scalar('model/non_embedding_var_norm',
                           tf.global_norm(self.non_embedding_params)),
         tf.summary.scalar('hparams/entropy_beta', self.entropy_hparam),
         tf.summary.scalar('hparams/topk_loss_hparam', self.config.topk_loss_hparam),
         tf.summary.scalar('hparams/learning_rate', self.learning_rate),
         tf.summary.scalar('model/trainable_var_norm',
                           tf.global_norm(self.trainable_variables)),
         tf.summary.scalar('loss/loss', self.loss),
         tf.summary.scalar('loss/entropy', self.entropy_loss_term),
         tf.summary.scalar('loss/policy', self.pi_loss_term),
         tf.summary.scalar('loss/offp', self.topk_loss_term)] +
        [tf.summary.scalar(
            'param_norms/' + _remove_prefix(scope_prefix + '/', p.name),
            tf.norm(p))
         for p in self.params] +
        [tf.summary.scalar(
            'grad_norms/' + _remove_prefix(scope_prefix + '/', p.name),
            tf.norm(g))
         for p, g in zip(self.params, self.grads)] +
        [tf.summary.scalar(
            'unclipped_grad_norms/' + _remove_prefix(scope_prefix + '/',
                                                     p.name),
            tf.norm(g))
         for p, g in zip(self.params, self.dense_unclipped_grads)])

    self.text_summary_placeholder = tf.placeholder(tf.string, shape=[])
    self.rl_text_summary_op = tf.summary.text('rl',
                                              self.text_summary_placeholder)

  def _rl_text_summary(self, session, step, npe, tot_r, num_steps, code, result):
    """Logs summary about a single episode and creates a text_summary for TB.

    Args:
      session: tf.Session instance.
      step: Global training step.
      npe: Number of programs executed so far.
      tot_r: Total reward.
      num_steps: Number of timesteps in the episode (i.e. code length).
      code: String representation of the code.
      result: Result of program execution

    Returns:
      Serialized text summary data for tensorboard.
    """
    if not code:
      code = ' '
    text = (
        'Tot R: **%.2f**;  Len: **%d**;  Result: **%s**\n\n'
        '\n\nCode: **`%s`**'
        % (tot_r, num_steps, result, code))
    text_summary = session.run(self.rl_text_summary_op,
                               {self.text_summary_placeholder: text})
    logger.info(
        'Step %d.\t NPE: %d\t Result: %s.\t Tot R: %.2f.\t Length: %d. \tProgram: %s',
        step, npe, result, tot_r, num_steps, code)
    return text_summary

  def _rl_reward_summary(self, total_rewards):
    """Create summary ops that report on episode rewards.

    Creates summaries for average, median, max, and min rewards in the batch.

    Args:
      total_rewards: Tensor of shape [batch_size] containing the total reward
          from each episode in the batch.

    Returns:
      tf.Summary op.
    """
    tr = np.asarray(total_rewards)
    reward_summary = tf.Summary(value=[
        tf.Summary.Value(
            tag='reward/avg',
            simple_value=np.mean(tr)),
        tf.Summary.Value(
            tag='reward/med',
            simple_value=np.median(tr)),
        tf.Summary.Value(
            tag='reward/max',
            simple_value=np.max(tr)),
        tf.Summary.Value(
            tag='reward/min',
            simple_value=np.min(tr))])
    return reward_summary

  def write_programs(self, session, inspiration_branch):
    """Generate Brainfuck programs with this language model"""

    # Sample new programs from the policy.
    # Note: batch size is constant. A full batch will be sampled, but not all
    # programs will be executed and added to the replay buffer. Those which
    # are not executed will be discarded and not counted.

    length_limit = f'code.str.len() < {self.config.timestep_limit}'
    self.inspiration_branch = inspiration_branch.query(length_limit)

    batch_actions, episode_lengths, log_probs = session.run(
        [self.sampled_batch.tokens,
         self.sampled_batch.episode_lengths, 
         self.sampled_batch.log_probs])

    if episode_lengths.size == 0:
      # This should not happen.
      logger.warn(
          'Shapes:\n'
          'batch_actions.shape: %s\n'
          'episode_lengths.shape: %s\n',
          batch_actions.shape, episode_lengths.shape)

    programs = make_dev_codebase()
    for actions, episode_length, action_log_probs in zip(
        batch_actions, episode_lengths, log_probs
      ):
      code = self.int_to_char(actions[:episode_length])
      if code[-1] == self.eos_char:
        code = code[:-1]

      logger.info(f'Wrote program: {code}')
      metrics = {'log_prob': np.choose(actions[:episode_length], 
                                       action_log_probs[:episode_length].T).sum()}
      programs.commit(code, metrics=metrics)

    return programs

  def accept_feedback(self, session, feedback_branch, train_op, global_step_op, return_gradients=False):
    """Improve language generation based on rewards from the real world

    Args:
      session: tf.Session instance.
      reinforcement: a Reinforcement object that summarizes the results of 
          recent program runs
      train_op: A TF op which will perform the gradient update. LMAgent does not
          own its training op, so that trainers can do distributed training
          and construct a specialized training op.
      global_step_op: A TF op which will return the current global step when
          run (should not increment it).
      return_gradients: If True, the gradients will be saved and returned from
          this method call. This is useful for testing.

    Returns:
      Results from the update step in a ReflectionResult namedtuple, including
      global step, global NPE, serialized summaries, and optionally gradients.
    """
    assert self.is_local
    assert len(feedback_branch) == self.config.batch_size

    if self.config.topk != 0:
      off_policy_branch = self.inspiration_branch.top_k('quality', self.config.topk)
      if len(off_policy_branch) > self.config.topk_batch_size:
        off_policy_branch = off_policy_branch.sample(self.config.topk_batch_size)
    else:
      off_policy_branch = make_codebase_like(self.inspiration_branch)
      
    if len(off_policy_branch):
      off_policy_target_lengths, off_policy_targets, _, _ = \
        self.process_episodes(off_policy_branch)
      
      offp_switch = 1
    else:
      off_policy_targets = [[0]]
      off_policy_target_lengths = [1]
      offp_switch = 0

    feedback_lengths, feedback_actions, feedback_targets, feedback_returns = \
      self.process_episodes(feedback_branch)

    # Do update for REINFORCE or REINFORCE + replay buffer.
    if self.num_replay_per_batch == 0:
      # Train with on-policy REINFORCE.
      num_programs_from_policy = self.config.batch_size
      
      # Process on-policy samples.
     
      batch_policy_multipliers = feedback_targets
      batch_emp_values = [[]]

      fetches = {
          'global_step': global_step_op,
          'program_count': self.program_count,
          'summaries': self.rl_summary_op,
          'train_op': train_op,
          'gradients': self.gradients_dict if return_gradients else self.no_op}

      fetched = session.run(
          fetches,
          {self.actions: feedback_actions,
          self.empirical_values: batch_emp_values,
          self.policy_multipliers: batch_policy_multipliers,
          self.adjusted_lengths: feedback_lengths,
          self.off_policy_targets: off_policy_targets,
          self.off_policy_target_lengths: off_policy_target_lengths,
          self.offp_switch: offp_switch})

      combined_adjusted_lengths = feedback_lengths
      combined_returns = feedback_returns
    else:
      # Train with REINFORCE + off-policy replay buffer by using importance
      # sampling.

      # Sample from experince replay buffer
      empty_replay_buffer = len(self.inspiration_branch) < self.num_replay_per_batch
      num_programs_from_replay_buff = (
          self.num_replay_per_batch if not empty_replay_buffer else 0)
      num_programs_from_policy = (
          self.config.batch_size - num_programs_from_replay_buff)
      if (not empty_replay_buffer) and num_programs_from_replay_buff:
        replay_branch = self.inspiration_branch.sample(num_programs_from_replay_buff)
        replay_lengths, replay_actions, replay_batch_targets, replay_returns = \
          self.process_episodes(replay_branch)

        replay_episode_actions = utils.stack_pad(replay_actions, pad_axes=0,
                                                 dtype=np.int32)

        # compute log probs for replay samples under current policy
        all_replay_log_probs, = session.run(
            [self.given_batch.log_probs],
            {self.actions: replay_episode_actions,
             self.adjusted_lengths: replay_lengths})

        replay_log_probs = [
            np.choose(replay_actions[i, :l], all_replay_log_probs[i, :l].T).sum()
            for i, l in enumerate(replay_lengths)]
        replay_branch['log_prob'] = replay_log_probs

        # Convert 2D array back into ragged 2D list.
        replay_policy_multipliers = [
            replay_batch_targets[i, :l]
            for i, l
            in enumerate(
                replay_lengths[:num_programs_from_replay_buff])]
      else:
        replay_lengths = None
        replay_actions = None
        replay_batch_targets = None
        replay_returns = None
        replay_policy_multipliers = None

      # Process on-policy samples.
      p = num_programs_from_policy
      
      on_policy_branch = feedback_branch.sample(p)
      adjusted_lengths, episode_actions, batch_targets, batch_returns = \
        self.process_episodes(on_policy_branch)
      batch_policy_multipliers = batch_targets
      batch_emp_values = [[]]
      on_policy_returns = batch_returns
        
      # On-policy episodes.
      if num_programs_from_policy:
        separate_actions = [
            episode_actions[i, :l]
            for i, l in enumerate(adjusted_lengths)]
        new_experiences = [
            (separate_actions[i],
            on_policy_branch['total_reward'][i],
            on_policy_branch['log_prob'][i], l)
            for i, l in enumerate(adjusted_lengths)]
        on_policy_policy_multipliers = [
            batch_policy_multipliers[i, :l]
            for i, l in enumerate(adjusted_lengths)]
        (on_policy_actions,
        _,  # rewards
        on_policy_log_probs,
        on_policy_adjusted_lengths) = zip(*new_experiences)
      else:
        new_experiences = []
        on_policy_policy_multipliers = []
        on_policy_actions = []
        on_policy_log_probs = []
        on_policy_adjusted_lengths = []

      if (not empty_replay_buffer) and num_programs_from_replay_buff:
        on_policy_branch['quality'] = 0
        # Look for new experiences in replay buffer. Assign weight if an episode
        # is in the buffer.
        on_policy_branch.replace(replay_branch)

      # Randomly select on-policy or off policy episodes to train on.
      combined_adjusted_lengths = list_(replay_lengths) + list_(on_policy_adjusted_lengths)
      combined_returns = utils.stack_pad(list_(replay_returns) + list_(on_policy_returns), pad_axes=0)
      combined_actions = utils.stack_pad(list_(replay_actions) + list_(on_policy_actions), pad_axes=0)
      combined_policy_multipliers = utils.stack_pad(list_(replay_policy_multipliers) + list_(on_policy_policy_multipliers),
                                                    pad_axes=0)

      # Importance adjustment. Naive formulation:
      # E_{x~p}[f(x)] ~= 1/N sum_{x~p}(f(x)) ~= 1/N sum_{x~q}(f(x) * p(x)/q(x)).
      # p(x) is the policy, and q(x) is the off-policy distribution, i.e. replay
      # buffer distribution. Importance weight w(x) = p(x) / q(x).

      # Instead of sampling from the replay buffer only, we sample from a
      # mixture distribution of the policy and replay buffer.
      # We are sampling from the mixture a*q(x) + (1-a)*p(x), where 0 <= a <= 1.
      # Thus the importance weight w(x) = p(x) / (a*q(x) + (1-a)*p(x))
      # = 1 / ((1-a) + a*q(x)/p(x)) where q(x) is 0 for x sampled from the
      #                             policy.
      # Note: a = self.replay_alpha
      if empty_replay_buffer:
        # The replay buffer is empty.
        # Do no gradient update this step. The replay buffer will have stuff in
        # it next time.
        combined_policy_multipliers *= 0
      elif not num_programs_from_replay_buff:
        combined_policy_multipliers = np.ones([len(combined_actions), 1],
                                              dtype=np.float32)
      else:
        # If a < 1 compute importance weights
        # importance weight
        # = 1 / [(1 - a) + a * exp(log(replay_weight / total_weight / p))]
        # = 1 / ((1-a) + a*q/p)
        combined_branch = Codebase(on_policy_branch.metrics,
                                   on_policy_branch.metadata,
                                   deduplication=False)
        combined_branch.merge(on_policy_branch)
        combined_branch.merge(replay_branch)

        importance_weights = compute_iw(combined_branch, self.replay_alpha)
        
        if self.config.iw_normalize:
          importance_weights *= (
              float(self.config.batch_size) / importance_weights.sum())
        combined_policy_multipliers *= importance_weights.reshape(-1, 1)

      # Train on replay batch, top-k MLE.
      assert self.program_count is not None
      fetches = {
          'global_step': global_step_op,
          'program_count': self.program_count,
          'summaries': self.rl_summary_op,
          'train_op': train_op,
          'gradients': self.gradients_dict if return_gradients else self.no_op}
      fetched = session.run(
          fetches,
          {self.actions: combined_actions,
          self.empirical_values: [[]],  # replay_emp_values,
          self.policy_multipliers: combined_policy_multipliers,
          self.adjusted_lengths: combined_adjusted_lengths,
          self.off_policy_targets: off_policy_targets,
          self.off_policy_target_lengths: off_policy_target_lengths,
          self.offp_switch: offp_switch})

    # Update program count.
    session.run(
        [self.program_count_add_op],
        {self.program_count_add_ph: num_programs_from_policy})

    # Update EMA baselines on the mini-batch which we just did traning on.
    for i, program in enumerate(feedback_branch['code']):
      episode_length = combined_adjusted_lengths[i]
      empirical_returns = combined_returns[i, :episode_length]
      for j in xrange(episode_length):
        # Update ema_baselines in place.
        self.ema_by_len[j] = (
            self.config.ema_baseline_decay * self.ema_by_len[j]
            + (1 - self.config.ema_baseline_decay) * empirical_returns[j])

    global_step = fetched['global_step']
    global_npe = fetched['program_count']
    core_summaries = fetched['summaries']
    summaries_list = [core_summaries]

    if num_programs_from_policy:
      s_i = 0
      text_summary = self._rl_text_summary(
          session,
          global_step,
          global_npe,
          feedback_branch['total_reward'][s_i],
          feedback_lengths[s_i], 
          feedback_branch['code'][s_i], 
          feedback_branch['result'][s_i])
      reward_summary = self._rl_reward_summary(feedback_branch['total_reward'])

      max_i = np.argmax(feedback_branch['total_reward'])
      max_tot_r = feedback_branch['total_reward'][max_i]
      if max_tot_r >= self.top_reward:
        if max_tot_r >= self.top_reward:
          self.top_reward = max_tot_r
        logger.info('Top code: r=%.2f, \t%s', max_tot_r, feedback_branch['code'][max_i])

      summaries_list += [text_summary, reward_summary]

      if self.do_iw_summaries and not empty_replay_buffer:
        # prob of replay samples under replay buffer sampling.
        total_weight = sum(replay_branch['quality'])
        norm_replay_weights = [
            w / total_weight
            for w in replay_branch['quality']]
        replay_iw = compute_iw(replay_branch, self.replay_alpha)
        on_policy_iw = compute_iw(on_policy_branch, self.replay_alpha)
        summaries_list.append(
            self._iw_summary(
                session, replay_iw, replay_log_probs, norm_replay_weights,
                on_policy_iw, on_policy_log_probs))

    return ReflectionResult(
        global_step=global_step,
        global_npe=global_npe,
        summaries_list=summaries_list,
        gradients_dict=fetched['gradients'])

  def process_episodes(self, reinforce_branch):
    """Compute REINFORCE targets.

    Treat a program as a reinforcement learning episode where a character
    is an action and 'quality' is the reward for the last character

    REINFORCE here takes the form:
    grad_t = grad[log(pi(a_t|c_t))*target_t]
    where c_t is context: i.e. RNN state or environment state (or both).

    Args:
      reinforce_branch: A Codebase with 'quality' metric
      baselines: Provide baselines for each timestep. This is a
          list (or indexable container) of length max_time. Note: baselines are
          shared across all episodes, which is why there is no batch dimension.
          It is up to the caller to update baselines accordingly.

    Returns:
      lengths: Length of every episode
      actions: List of actions for every episode
      targets: REINFORCE targets for each episode and timestep. A numpy
          array of shape [batch_size, max_sequence_length].
      returns: Returns computed for each episode and timestep. This is for
          reference, and is not used in the REINFORCE gradient update (but was
          used to compute the targets). A numpy array of shape
          [batch_size, max_sequence_length].
    """
    assert 'quality' in reinforce_branch.metrics, 'Reinforcement has to contain a quality metric'
    assert len(reinforce_branch)

    lengths = [len(code) for code in reinforce_branch['code']]
    baselines = self.ema_by_len

    # We only reward the last character
    # Everything else is a preparation for dat final punch
    action_rewards = utils.stack_pad(
      [[0] * (episode_length - 1) + [episode_reward]
      for episode_reward, episode_length in 
      zip(reinforce_branch['quality'], lengths)],
      pad_axes=0
    )

    num_programs = len(action_rewards)
    
    batch_returns = [None] * num_programs
    batch_targets = [None] * num_programs
    for i in xrange(num_programs):
      episode_length = lengths[i]
      # Compute target for each timestep.
      #    target_t = R_t - baselines[t]
      #    where `baselines` are provided.
      # In practice we use a more generalized formulation of advantage. See docs
      # for `discounted_advantage_and_rewards`.
      
      # Compute return for each timestep. See section 3 of
      # https://arxiv.org/pdf/1602.01783.pdf
      assert baselines is not None
      empirical_returns = rollout_lib.discount(action_rewards[i], gamma=1.0)
      targets = [None] * episode_length
      for j in xrange(episode_length):
        targets[j] = empirical_returns[j] - baselines[j]
      batch_returns[i] = empirical_returns
      batch_targets[i] = targets

    actions = utils.stack_pad(
              [self.char_to_int(code)
                for code in reinforce_branch['code']], 
                pad_axes=0)
    actions = np.array(actions, dtype=int)

    returns = utils.stack_pad(batch_returns, 0)
    if num_programs:
      targets = utils.stack_pad(batch_targets, 0)
    else:
      targets = np.array([], dtype=np.float32)

    return (lengths, actions, targets, returns)

def compute_iw(codebase, replay_alpha):
    """Compute importance weights for a batch of episodes.

    The codebase has to contain a 'quality' and 'log_prob' metrics

    Returns:
      Numpy array of shape [batch_size] containing the importance weight for
      each episode in the batch.
    """
    total_replay_weight = sum(codebase['quality'])
    if total_replay_weight == 0:
      # This happens when:
      # - codebase is empty
      # or
      # - all replay weights are zero due to floating point limitations 
      # replay weight are exponents and sometimes exp(minus a lot) is rounded to 0
      n = len(codebase)
      return np.ones(n) / n
    else:
      try:
        log_total_replay_weight = log(total_replay_weight)

        # importance weight
        # = 1 / [(1 - a) + a * exp(log(replay_weight / total_weight / p))]
        # = 1 / ((1-a) + a*q/p)
        a = float(replay_alpha)
        a_com = 1.0 - a  # compliment of a

        importance_weights = np.asarray(
            [1.0 / (a_com
                    + a * exp((log(replay_weight) - log_total_replay_weight)
                              - log_p))
            if replay_weight > 0 else 1.0 / a_com
            for log_p, replay_weight
            in zip(codebase['log_prob'], codebase['quality'])])
        return importance_weights
      except OverflowError:
        # This Softmax is too close for the CPU to handle
        # So it's safe to turn it into just max
        importance_weights = np.zeros(len(codebase))
        importance_weights[np.argmax(codebase['quality'])] = 1
      return importance_weights
