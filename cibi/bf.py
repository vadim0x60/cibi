from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Extended BrainF**k interpreter.

Language info: https://en.wikipedia.org/wiki/Brainfuck
+ 0 to set memory to null value
+ ! to write to the beginning of the stack
"""

from cibi.agent import Agent, ActionError

from collections import namedtuple
import gym.spaces as s
import numpy as np
import time

from typing import TYPE_CHECKING

ExecutionSnapshot = namedtuple(
    'ExecutionSnapshot',
    ['codeptr', 'codechar', 'memptr', 'memval', 'memory', 'action_stack', 'state'])

class Result(object):
  SUCCESS = 'success'
  STEP_LIMIT = 'step-limit'
  SYNTAX_ERROR = 'syntax-error'
  KILLED = 'killed'

class State(object):
  NOT_STARTED = 'not-started'
  EXECUTING = 'executing'
  AWAITING_INPUT = 'awaiting-input'
  FINISHED = 'finished'

CHARS = ['>', '<', '+', '-', '[', ']', '.', ',', '!', '0']
BF_EOS_INT = 0  # Also used as SOS (start of sequence).
BF_EOS_CHAR = TEXT_EOS_CHAR = '_'
BF_INT_TO_CHAR = [BF_EOS_CHAR] + CHARS
BF_CHAR_TO_INT = dict([(c, i) for i, c in enumerate(BF_INT_TO_CHAR)])

def buildbracemap(code):
  """Build jump map.

  Args:
    code: List or string or BF chars.

  Returns:
    bracemap: dict mapping open and close brace positions in the code to their
        destination jumps. Specifically, positions of matching open/close braces
        if they exist.
    correct_syntax: True if all braces match. False if there are unmatched
        braces in the code. Even if there are unmatched braces, a bracemap will
        be built, and unmatched braces will map to themselves.
  """
  bracestack, bracemap = [], {}

  correct_syntax = True
  for position, command in enumerate(code):
    if command == '[':
      bracestack.append(position)
    if command == ']':
      if not bracestack:  # Unmatched closing brace.
        bracemap[position] = position  # Don't jump to any position.
        correct_syntax = False
        continue
      start = bracestack.pop()
      bracemap[start] = position
      bracemap[position] = start
  if bracestack:  # Unmatched opening braces.
    for pos in bracestack:
      bracemap[pos] = pos  # Don't jump to any position.
      correct_syntax = False
  return bracemap, correct_syntax

class ProgramFinishedError(ActionError):
  def __init__(self, code, program_result):
    msg = f'Trying to execute program {code} that has finished with {program_result}'
    super().__init__(msg, program_result)

class TuringMemoryWriter:
  def __init__(self, observation_space, discretization_steps=32):
    if type(observation_space) in (s.Discrete, s.MultiDiscrete, s.MultiBinary):
      self.discretization_steps = 1
    elif type(observation_space) == s.Box:
      self.discretization_steps = discretization_steps
    else:
      raise NotImplementedError('Only Discrete, MultiDiscrete, MultiBinary and Box spaces are supported')
    
  def discretize(self, value):
    return int(value * self.discretization_steps)
    
  def write(self, memory, cellptr, inp):
    value_ptr = cellptr
    for _, value in np.ndenumerate(inp):
      discrete_value = self.discretize(value)

      # extend or rewrite
      if value_ptr == len(memory): 
        memory.append(discrete_value)
      else:
        memory[value_ptr] = discrete_value
      value_ptr += 1

class ActionSampler:
  def __init__(self, action_space, discretization_steps=32):
    space_type = type(action_space)
    self.sample_shape = action_space.shape
    self.discretization_steps = discretization_steps

    if space_type == s.Discrete:
      self.lower_bound = 0
      self.upper_bound = action_space.n - 1
      self.bounded_below = True
      self.bounded_above = True

      # Optional, but mapping 0-15 to 0 and 16-31 to 1 
      # in a binary case is an unnecessary complication
      # So let's not
      self.discretization_steps = action_space.n

    elif space_type == s.MultiDiscrete:
      self.upper_bound = action_space.nvec
      self.lower_bound = np.zeros_like(self.upper_bound)
      self.bounded_above = np.ones_like(self.upper_bound)
      self.bounded_below = np.ones_like(self.lower_bound)

    elif space_type == s.MultiBinary:
      self.lower_bound = np.zeros(action_space.n)
      self.upper_bound = np.ones(action_space.n)
      self.bounded_above = np.ones_like(self.upper_bound)
      self.bounded_below = np.ones_like(self.lower_bound)

    elif space_type == s.Box:
      self.lower_bound = action_space.lower_bound
      self.upper_bound = action_space.upper_bound
      self.bounded_above = action_space.bounded_above
      self.bounded_below = action_space.bounded_below

    else:
      raise NotImplementedError('Only Discrete, MultiDiscrete, MultiBinary and Box spaces are supported')

  def undiscretize(self, discrete_actions):
    double_bounded = self.bounded_below * self.bounded_above
    unbounded = (1 - self.bounded_below) * (1 - self.bounded_above)

    actions = double_bounded * np.mod(discrete_actions, self.discretization_steps)
    actions += self.bounded_below * self.lower_bound
    actions += self.bounded_above * (1 - self.bounded_below) * self.upper_bound
    actions += (self.bounded_below - self.bounded_above) * np.abs(actions)
    actions += unbounded * actions

    return actions

  def sample(self, action_stack):
    sample_size = int(np.prod(self.sample_shape))

    # like pop(), but for many elements
    sample = action_stack[-sample_size:]
    del action_stack[-sample_size:]

    if sample_size == 1:
      sample = sample[0]
    else:
      sample = np.array(sample).reshape(self.sample_shape)

    return self.undiscretize(sample)
    
class Executable(Agent):
  def __init__(self, code, memory_writer, action_sampler,
               init_memory=None, base=256, null_value=0,
               max_steps=2 ** 20, require_correct_syntax=True, debug=False,
               cycle = False):
    self.code = code
    code = list(code)
    self.bracemap, correct_syntax = buildbracemap(code)  # will modify code list
    if len(code) == 0:
      # Empty programs are a very easy way for a lazy-bum developer
      # to avoid negative reinforcement for syntax errors
      # Not so fast, lazy-bum developers
      correct_syntax = False

    self.is_valid = correct_syntax or not require_correct_syntax

    self.memory_writer = memory_writer
    self.action_sampler = action_sampler
    
    self.init_memory = init_memory
    self.max_steps = max_steps
    self.debug = debug
    self.base = base
    self.null_value = null_value
    self.cycle = cycle

    self.init()

  def init(self):
    self.program_trace = [] if self.debug else None
    self.codeptr, self.cellptr = 0, 0
    self.steps = 0
    self.cells = list(self.init_memory) if self.init_memory else [0]
    self.action_stack = []

    if not self.is_valid:
      self.state = State.FINISHED
      self.result = Result.SYNTAX_ERROR
    else:
      self.state = State.NOT_STARTED
      self.result = None

  def record_snapshot(self, command):
    if self.debug:
      # Add step to program trace.
      self.program_trace.append(ExecutionSnapshot(
          codeptr=self.codeptr, codechar=command, memptr=self.cellptr,
          memval=self.cells[self.cellptr], memory=list(self.cells),
          state=self.state, action_stack=self.action_stack))

  def value(self):
    return self.value_estimate

  def done(self):
    if self.state != State.FINISHED:
      self.state = State.FINISHED
      self.result = Result.KILLED

  def step(self):
    if self.state == State.FINISHED:
      raise ProgramFinishedError(self.code, self.result)
    if self.state == State.AWAITING_INPUT:
      return

    self.state = State.EXECUTING

    if self.max_steps is not None and self.steps >= self.max_steps:
      self.result = Result.STEP_LIMIT
      self.state = State.FINISHED
      return

    if self.codeptr == len(self.code):
      if self.cycle:
        self.codeptr = -1
        self.state = State.AWAITING_INPUT
        self.record_snapshot(',')
        return
      else:
        self.state = State.FINISHED
        self.result = Result.SUCCESS
        return

    command = self.code[self.codeptr]
    self.record_snapshot(command)

    if command == '>':
      self.cellptr += 1
      if self.cellptr == len(self.cells): self.cells.append(self.null_value)

    if command == '<':
      self.cellptr = 0 if self.cellptr <= 0 else self.cellptr - 1

    if command == '+':
      self.cells[self.cellptr] = self.cells[self.cellptr] + 1 if self.cells[self.cellptr] < (self.base - 1) else 0

    if command == '-':
      self.cells[self.cellptr] = self.cells[self.cellptr] - 1 if self.cells[self.cellptr] > 0 else (self.base - 1)

    if command == '0':
      self.cells[self.cellptr] = self.null_value

    if command == '[' and self.cells[self.cellptr] == 0: self.codeptr = self.bracemap[self.codeptr]
    if command == ']' and self.cells[self.cellptr] != 0: self.codeptr = self.bracemap[self.codeptr]

    if command == '.': self.action_stack.insert(0, self.cells[self.cellptr])
    if command == '!': self.action_stack.append(self.cells[self.cellptr])

    if command == ',':
      self.state = State.AWAITING_INPUT
      self.record_snapshot(command)
      return

    self.codeptr += 1
    self.steps += 1

  def execute(self):
    self.step()

    while self.state == State.EXECUTING:
      self.step()

  def input(self, inp):
    while self.state != State.AWAITING_INPUT:
      self.execute()

    self.state = State.EXECUTING
    self.record_snapshot(',')

    self.memory_writer.write(self.cells, self.cellptr, inp)

    self.codeptr += 1
    self.steps += 1

  def act(self):
    try:
      return self.action_sampler.sample(self.action_stack)
    except IndexError:
      return self.null_value

class Program:
    code = ""
    log_probs = None
    value_estimate = None
    cycle = False

    def compile(self, *args, **kwargs):
        executable = Executable(self.code, *args, **kwargs)
        executable.log_probs = self.log_probs
        executable.value_estimate = self.value_estimate
        return executable

    def __init__(self, code, log_probs=None, value_estimate=None):
      self.code = code
      self.log_probs = log_probs
      self.value_estimate = value_estimate