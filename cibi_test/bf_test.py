from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.bf."""

import tensorflow as tf
import gym.spaces as s
from cibi import bf  # brain coder
from cibi import bf_io
from cibi.bf import Executable, ProgramFinishedError

action_space = s.Discrete(1024)
observation_space = s.Discrete(1024)

observation_discretizer = bf_io.ObservationDiscretizer(observation_space, history_length=None)
action_sampler = bf_io.ActionSampler(action_space)

def shorten(seq, size_limit=20, head_size=5, tail_size=5):
  if len(seq) < size_limit:
    return seq
  else:
    return seq[:head_size] + ['...'] + seq[-tail_size:]

def evaluate(code, **kwargs):
  """Adapter between our agent approach and Google's seq2seq approach"""

  try:
    input_buffer = kwargs['input_buffer']
    del kwargs['input_buffer']
  except KeyError:
    input_buffer = []

  if 'debug' not in kwargs:
    # This are unit tests! Debug mode on by default
    kwargs['debug'] = True

  agent = Executable(code, observation_discretizer=observation_discretizer, 
                           action_sampler=action_sampler, 
                           **kwargs)

  for x in input_buffer:
    agent.input(x)

  if agent.state != bf.State.FINISHED:
    agent.execute()

  while agent.state == bf.State.AWAITING_INPUT:
    agent.input(agent.null_value)

  if agent.state != bf.State.FINISHED:
    agent.execute()

  if kwargs['debug']:
    print(shorten(agent.program_trace))
    print(agent.result)

  return agent

class BfTest(tf.test.TestCase):

  def assertCorrectOutput(self, target_output, agent):
    output = list(reversed(agent.action_stack))
    self.assertEqual(target_output, output)
    self.assertEqual(bf.State.FINISHED, agent.state)
    self.assertEqual(bf.Result.SUCCESS, agent.result)

  def testBasicOps(self):
    self.assertCorrectOutput(
        [3, 1, 2],
        evaluate('+++.--.+.'))
    self.assertCorrectOutput(
        [1, 1, 2],
        evaluate('+.<.>++.'))
    self.assertCorrectOutput(
        [0],
        evaluate('+,.'))
        
  def testUnmatchedBraces(self):
    self.assertCorrectOutput(
        [3, -4, 1],
        evaluate('+++.]]]]>----.[[[[[>+.',
                    input_buffer=[],
                    require_correct_syntax=False))

    agent = evaluate(
        '+++.]]]]>----.[[[[[>+.',
        input_buffer=[],
        require_correct_syntax=True)
    self.assertEqual([], agent.action_stack)
    self.assertEqual(bf.Result.SYNTAX_ERROR, agent.result)

  def testMaxSteps(self):
    agent = evaluate('+.[].', input_buffer=[], max_steps=100)
    output = list(reversed(agent.action_stack))
    self.assertEqual([1], output)
    self.assertEqual(bf.Result.STEP_LIMIT, agent.result)

    agent = evaluate('+.[-].', input_buffer=[], max_steps=100)
    output = list(reversed(agent.action_stack))
    self.assertEqual([1, 0], output)
    self.assertEqual(bf.Result.SUCCESS, agent.result)

  def testOutputMemory(self):
    agent = evaluate('+>++>+++>++++.', input_buffer=[])
    output = list(reversed(agent.action_stack))
    self.assertEqual([4], output)
    self.assertEqual(bf.Result.SUCCESS, agent.result)
    self.assertEqual([1, 2, 3, 4], agent.cells)

  def testProgramTrace(self):
    es = bf.ExecutionSnapshot
    agent = evaluate('[.>,].', input_buffer=[2, 1], debug=True)
    self.assertEqual(
        [es(codeptr=1, codechar=',', memptr=0, memval=2, memory=[2], action_stack=[], state='executing'), 
         es(codeptr=1, codechar='.', memptr=0, memval=2, memory=[2], action_stack=[], state='executing'), 
         es(codeptr=2, codechar='>', memptr=0, memval=2, memory=[2], action_stack=[2], state='executing'), 
         es(codeptr=3, codechar=',', memptr=1, memval=0, memory=[2, 0], action_stack=[2], state='executing'), 
         es(codeptr=3, codechar=',', memptr=1, memval=0, memory=[2, 0], action_stack=[2], state='awaiting-input'), 
         es(codeptr=4, codechar=',', memptr=1, memval=1, memory=[2, 1], action_stack=[2], state='executing'), 
         es(codeptr=4, codechar=']', memptr=1, memval=1, memory=[2, 1], action_stack=[2], state='executing'), 
         es(codeptr=1, codechar='.', memptr=1, memval=1, memory=[2, 1], action_stack=[2], state='executing'), 
         es(codeptr=2, codechar='>', memptr=1, memval=1, memory=[2, 1], action_stack=[1, 2], state='executing'), 
         es(codeptr=3, codechar=',', memptr=2, memval=0, memory=[2, 1, 0], action_stack=[1, 2], state='executing'), 
         es(codeptr=3, codechar=',', memptr=2, memval=0, memory=[2, 1, 0], action_stack=[1, 2], state='awaiting-input'), 
         es(codeptr=4, codechar=',', memptr=2, memval=0, memory=[2, 1, 0], action_stack=[1, 2], state='executing'), 
         es(codeptr=4, codechar=']', memptr=2, memval=0, memory=[2, 1, 0], action_stack=[1, 2], state='executing'), 
         es(codeptr=5, codechar='.', memptr=2, memval=0, memory=[2, 1, 0], action_stack=[1, 2], state='executing')],
        agent.program_trace)

  def testStreamDiscretizer(self):
    discretize = bf_io.StreamDiscretizer([0, 1])
    self.assertEqual([discretize(x) for x in [-1, 0, 0.5, 1, 1.5]], [0, 1, 1, 2, 2])

  def testFluidStreamDiscretizer(self):
    discretize = bf_io.FluidStreamDiscretizer(bin_count=3, history_length=2)
    print(discretize.thresholds)
    discretize(5)
    discretize(6)
    print(discretize.thresholds)
    self.assertEqual(discretize(5.5), 1)


if __name__ == '__main__':
  tf.test.main()
