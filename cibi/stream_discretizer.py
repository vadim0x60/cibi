## \sec{fluid-discretizer} in FIXME paper

from sortedcollections import ValueSortedDict
import numpy as np

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