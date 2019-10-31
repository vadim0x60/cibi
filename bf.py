from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Extended BrainF**k interpreter.

Language info: https://en.wikipedia.org/wiki/Brainfuck
+ 0 to set memory to null value
+ ! to write to the beginning of the stack
"""

from collections import namedtuple
import time

ExecutionSnapshot = namedtuple(
    'ExecutionSnapshot',
    ['codeptr', 'codechar', 'memptr', 'memval', 'memory', 'action_stack', 'state'])

class Result(object):
  SUCCESS = 'success'
  STEP_LIMIT = 'step-limit'
  SYNTAX_ERROR = 'syntax-error'

class State(object):
  NOT_STARTED = 'not-started'
  EXECUTING = 'executing'
  AWAITING_INPUT = 'awaiting-input'
  FINISHED = 'finished'

CHARS = INT_TO_CHAR = ['>', '<', '+', '-', '[', ']', '.', ',', '!', '0']
CHAR_TO_INT = dict([(c, i) for i, c in enumerate(INT_TO_CHAR)])

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

class ProgramFinishedError(RuntimeError):
  pass

class BrainfuckAgent():
  def __init__(self, code, init_memory=None, base=256, null_value=0,
               max_steps=2 ** 20, require_correct_syntax=True, debug=False):
    code = list(code)
    self.bracemap, correct_syntax = buildbracemap(code)  # will modify code list

    self.code = code
    self.max_steps = max_steps
    self.debug = debug
    self.base = base
    self.null_value = null_value

    self.program_trace = [] if debug else None
    self.codeptr, self.cellptr = 0, 0
    self.steps = 0
    self.cells = list(init_memory) if init_memory else [0]
    self.action_stack = []

    if require_correct_syntax and not correct_syntax:
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

  def execute(self):
    if self.state == State.FINISHED:
      raise ProgramFinishedError
    if self.state == State.AWAITING_INPUT:
      return

    self.state = State.EXECUTING

    while self.codeptr < len(self.code):
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

      if self.max_steps is not None and self.steps >= self.max_steps:
        self.result = Result.STEP_LIMIT
        self.state = State.FINISHED
        return

    self.record_snapshot(command)

    self.state = State.FINISHED
    self.result = Result.SUCCESS

  def input(self, inp):
    while self.state != State.AWAITING_INPUT:
      self.execute()

    self.state = State.EXECUTING
    self.record_snapshot(',')

    self.cells[self.cellptr] = inp

    self.codeptr += 1
    self.steps += 1


  def act(self):
    try:
      return self.action_stack.pop()
    except IndexError:
      return self.null_value