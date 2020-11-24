from sortedcollections import ValueSortedDict
import numpy as np
from collections import namedtuple
import gym.spaces as s

DEFAULT_STEPS = 5

ObservationSnapshot = namedtuple(
  'ObservationSnapshot',
  ['raw_observaton', 'memory_update'])

ActionSnapshot = namedtuple(
  'ActionSnapshot',
  ['action_stack', 'raw_action', 'action_taken']
)

class StreamDiscretizer():
  def __init__(self, thresholds):
    self.thresholds = thresholds
    self.offset = - len(self.thresholds) / 2
    self.saturated = True
    
  def __call__(self, value):
    return np.digitize(value, self.thresholds)

def floor(x):
  return np.floor(x).astype(int)

class DummyStreamDiscretizer():
  def __init__(self, offset=0):
    self.thresholds = []
    self.offset = offset
    self.saturated = True

  def __call__(self, value):
    return value

class FluidStreamDiscretizer():
  def __init__(self, bin_count, history_length):
    self.history = ValueSortedDict()
    self.step = 0
    self.thresholds = np.linspace(0, 1, bin_count - 1)
    self.history_length = history_length
    self.saturated = False

  def __call__(self, value):
    self.step += 1
    self.history[self.step] = value

    values = np.array(self.history.values())
    bin_count = len(self.thresholds) + 1
    self.thresholds = values[[floor(idx * len(self.history) / bin_count) for idx in range(1, bin_count)]]

    try:
      del self.history[self.step - self.history_length]
      self.saturated = True
    except KeyError:
      pass

    return np.digitize(value, self.thresholds)

def burn_in(env, agent, observation_discretizer, action_sampler):
  episode_count = 0
  if observation_discretizer.is_fluid():
    while not observation_discretizer.is_saturated():
        agent.attend_gym(env, render=False)
        episode_count += 1

    if observation_discretizer.debug:
        observation_discretizer.trace = []
    if action_sampler.debug:
        action_sampler.trace = []
  return episode_count

class ObservationDiscretizer():
  def __init__(self, observation_space, history_length, thresholds=None, debug=False, force_fluid=False):
    thresholds = np.array(thresholds)

    if type(observation_space) == s.Box:
      if thresholds:
        if len(thresholds.shape) == 1:
          self.discretizers = [StreamDiscretizer(thresholds) 
                               for _ in range(observation_space.shape[0])]
        else:
          assert thresholds.shape[:-1] == observation_space.shape
          self.discretizers = [StreamDiscretizer(t) for t in thresholds.reshape(-1)]
      else:
        self.discretizers = []

        lower_bounds = observation_space.low.reshape(-1)
        upper_bounds = observation_space.high.reshape(-1)

        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
          # Observation space object has bounded_below and bounded_above
          # attributes, but those are unreliable and in many environments
          # just always set to True
          bounded_below = (lower_bound > float('-3.4e+38'))
          bounded_above = (upper_bound < float('3.4e+38'))

          if (not force_fluid) and bounded_below and bounded_above:
            thresholds = np.linspace(lower_bound, upper_bound, num=DEFAULT_STEPS+1)[1:-1]
            discretizer = StreamDiscretizer(thresholds)
          else:
            discretizer = FluidStreamDiscretizer(bin_count=DEFAULT_STEPS, 
                                                 history_length=history_length) 

          self.discretizers.append(discretizer)            
    elif type(observation_space) in (s.Discrete, s.MultiDiscrete, s.MultiBinary):
      self.discretizers = []
    else:
      msg = f'{type(observation_space)} observation spaces not supported'
      raise NotImplementedError(msg)

    if debug:
      self.trace = []

    self.debug = debug

  def discretize(self, observation):
    if self.discretizers:
      observation = np.array(observation)
      discretized = [_discretize(feature) for _discretize, feature in zip(self.discretizers, observation.reshape(-1))]
      discretized = np.array(discretized).reshape(observation.shape)    
    else:
      discretized = observation
    
    if self.debug:
      self.trace.append(ObservationSnapshot(raw_observaton=observation,
                                            memory_update=discretized))
    
    return discretized

  def get_thresholds(self):
    return [d.thresholds for d in self.discretizers]

  def is_fluid(self):
    return any(type(d) == FluidStreamDiscretizer for d in self.discretizers)

  def is_saturated(self):
    return all(d.saturated for d in self.discretizers)

  def __call__(self, observation):
    return self.discretize(observation)

class ActionSampler:
  def __init__(self, action_space, discretization_steps=DEFAULT_STEPS, default_action=None, debug=False):
    space_type = type(action_space)
    self.sample_shape = action_space.shape
    self.discretization_steps = discretization_steps
    self.debug = debug
    self.just_a_number = False

    if debug:
      self.trace = []

    if space_type == s.Discrete:
      self.lower_bound = np.array([0])
      self.upper_bound = np.array([action_space.n - 1])
      self.bounded_below = np.array([True])
      self.bounded_above = np.array([True])
      self.default_action = 0
      self.sample_shape = (1,)
      self.just_a_number = True

      # Optional, but mapping 0-15 to 0 and 16-31 to 1 
      # in a binary case is an unnecessary complication
      # So let's not
      self.discretization_steps = action_space.n

    elif space_type == s.MultiDiscrete:
      self.upper_bound = action_space.nvec
      self.lower_bound = np.zeros_like(self.upper_bound)
      self.bounded_above = np.full(self.upper_bound.shape, True)
      self.bounded_below = np.full(self.lower_bound.shape, True)
      self.default_action = self.lower_bound

    elif space_type == s.MultiBinary:
      self.lower_bound = np.zeros(action_space.n)
      self.upper_bound = np.ones(action_space.n)
      self.bounded_above = np.full(self.upper_bound.shape, True)
      self.bounded_below = np.full(self.upper_bound.shape, True)
      self.default_action = self.lower_bound

    elif space_type == s.Box:
      self.lower_bound = action_space.low
      self.upper_bound = action_space.high
      self.bounded_above = action_space.bounded_above
      self.bounded_below = action_space.bounded_below
      self.default_action = np.zeros(self.sample_shape, dtype=np.float)

    else:
      raise NotImplementedError('Only Discrete, MultiDiscrete, MultiBinary and Box spaces are supported')

    self.lower_bound = self.lower_bound.reshape(-1)
    self.upper_bound = self.upper_bound.reshape(-1)
    self.bounded_below = self.bounded_below.reshape(-1)
    self.bounded_above = self.bounded_above.reshape(-1)

    # Override defaults
    if default_action is not None:
      self.default_action = default_action

  def undiscretize_action(self, idx, discrete_action):
    # See the paper for formula and explanation
    # TODO: Add link to arxiv
    if not self.bounded_below[idx] and self.bounded_above[idx]:
      action = discrete_action / self.discretization_steps

    if self.bounded_below[idx] and not self.bounded_above[idx]:
      action = discrete_action / (self.discretization_steps - 1)
      action = self.lower_bound[idx] + np.abs(action - self.lower_bound[idx])

    if not self.bounded_below[idx] and self.bounded_above[idx]:
      action = discrete_action / (self.discretization_steps - 1)
      action = self.upper_bound[idx] + np.abs(self.upper_bound[idx] - action)

    if self.bounded_below[idx] and self.bounded_above[idx]:
      action = np.mod(discrete_action, self.discretization_steps) / (self.discretization_steps - 1)
      action = self.lower_bound[idx] + action * (self.upper_bound[idx] - self.lower_bound[idx])

    return action

  def sample(self, action_stack):
    sample_size = int(np.prod(self.sample_shape))

    # like pop(), but for many elements
    raw_action = None
    action = self.default_action

    if len(action_stack) >= sample_size:
      action = raw_action = action_stack[-sample_size:]
      del action_stack[-sample_size:]

      action = [self.undiscretize_action(idx, a) for idx, a in enumerate(action)]
      
      if self.just_a_number:
        action = int(action[0])
      else:
        action = np.array(action).reshape(self.sample_shape)
    
    if self.debug:
      self.trace.append(ActionSnapshot(action_stack=action_stack,
                                       raw_action=raw_action,
                                       action_taken=action))

    return action

  def __call__(self, action_stack):
    return self.sample(action_stack)