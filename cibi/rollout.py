from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Utilities related to computing training batches from episode rollouts.

Implementations here are based on code from Open AI:
https://github.com/openai/universe-starter-agent/blob/master/a3c.py.
"""

from collections import namedtuple
import numpy as np
import scipy.signal

from cibi import utils


class Rollout(object):
  """Holds a rollout for an episode.

  A rollout is a record of the states observed in some environment and actions
  taken by the agent to arrive at those states. Other information includes
  rewards received after each action, values estimated for each state, whether
  the rollout concluded the episide, and total reward received. Everything
  should be given in time order.

  At each time t, the agent sees state s_t, takes action a_t, and then receives
  reward r_t. The agent may optionally estimate a state value V(s_t) for each
  state.

  For an episode of length T:
  states = [s_0, ..., s_(T-1)]
  actions = [a_0, ..., a_(T-1)]
  rewards = [r_0, ..., r_(T-1)]
  values = [V(s_0), ..., V(s_(T-1))]

  Note that there is an extra state s_T observed after taking action a_(T-1),
  but this is not included in the rollout.

  Rollouts have an `terminated` attribute which is True when the rollout is
  "finalized", i.e. it holds a full episode. terminated will be False when
  time steps are still being added to it.
  """

  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.total_reward = 0.0
    self.terminated = False

  def add(self, state, action, reward, terminated=False):
    """Add the next timestep to this rollout.

    Args:
      state: The state observed at the start of this timestep.
      action: The action taken after observing the given state.
      reward: The reward received for taking the given action.
      terminated: Whether this timestep ends the episode.

    Raises:
      ValueError: If this.terminated is already True, meaning that the episode
          has already ended.
    """
    if self.terminated:
      raise ValueError(
          'Trying to add timestep to an already terminal rollout.')
    self.states += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.terminated = terminated
    self.total_reward += reward

  def add_many(self, states, actions, rewards, terminated=False):
    """Add many timesteps to this rollout.

    Arguments are the same as `add`, but are lists of equal size.

    Args:
      states: The states observed.
      actions: The actions taken.
      rewards: The rewards received.
      terminated: Whether this sequence ends the episode.

    Raises:
      ValueError: If the lengths of all the input lists are not equal.
      ValueError: If this.terminated is already True, meaning that the episode
          has already ended.
    """
    if len(states) != len(actions):
      raise ValueError(
          'Number of states and actions must be the same. Got %d states and '
          '%d actions' % (len(states), len(actions)))
    if len(states) != len(rewards):
      raise ValueError(
          'Number of states and rewards must be the same. Got %d states and '
          '%d rewards' % (len(states), len(rewards)))
    if self.terminated:
      raise ValueError(
          'Trying to add timesteps to an already terminal rollout.')
    self.states += states
    self.actions += actions
    self.rewards += rewards
    self.terminated = terminated
    self.total_reward += sum(rewards)

  def extend(self, other):
    """Append another rollout to this rollout."""
    assert not self.terminated
    self.states.extend(other.states)
    self.actions.extend(other.actions)
    self.rewards.extend(other.rewards)
    self.terminated = other.terminated
    self.total_reward += other.total_reward

  def __len__(self):
    count = len(self.states)
    assert count == len(self.actions)
    assert count == len(self.states)
    return count


def discount(x, gamma):
  """Returns discounted sums for each value in x, with discount factor gamma.

  This can be used to compute the return (discounted sum of rewards) at each
  timestep given a sequence of rewards. See the definitions for return and
  REINFORCE in section 3 of https://arxiv.org/pdf/1602.01783.pdf.

  Let g^k mean gamma ** k.
  For list [x_0, ..., x_N], the following list of discounted sums is computed:
  [x_0 + g^1 * x_1 + g^2 * x_2 + ... g^N * x_N,
   x_1 + g^1 * x_2 + g^2 * x_3 + ... g^(N-1) * x_N,
   x_2 + g^1 * x_3 + g^2 * x_4 + ... g^(N-2) * x_N,
   ...,
   x_(N-1) + g^1 * x_N,
   x_N]

  Args:
    x: List of numbers [x_0, ..., x_N].
    gamma: Float between 0 and 1 (inclusive). This is the discount factor.

  Returns:
    List of discounted sums.
  """
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]