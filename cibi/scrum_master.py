from cibi.agent import Agent
from cibi import bf
from cibi import utils

import numpy as np
from collections import namedtuple

Reinforcement = namedtuple(
    'Reinforcement',
    ['episode_count', 'episode_lengths', 'episode_code_strings', 
     'episode_actions', 'action_rewards', 'episode_rewards', 'episode_values', 'episode_results'])

def programs_as_rl_episodes(programs): 
  episode_rewards = np.array(
    [sum(program.rewards)
     for program in programs]
  )
  episode_lengths = np.array(
    [len(program.code) 
     for program in programs]
  )
  # We only reward the last character
  # Everything else is a preparation for dat final punch
  action_rewards = utils.stack_pad(
    [[0] * (episode_length - 1) + [episode_reward]
     for episode_reward, episode_length in zip(episode_rewards, episode_lengths)],
     pad_axes=0
  )
  action_rewards = np.array(action_rewards)
  
  episode_values = np.array(
    [program.value_estimate
     for program in programs]
  )
  episode_code_strings = [program.code
                          for program in programs]
  episode_actions = utils.stack_pad(
                    [[bf.BF_CHAR_TO_INT[c] for c in program.code] 
                     for program in programs], 
                     pad_axes=0)
  episode_actions = np.array(episode_actions)
  episode_results = [program.result
                     for program in programs]
  
  return Reinforcement(episode_count = len(episode_lengths),
                       episode_lengths=episode_lengths,
                       action_rewards=action_rewards,
                       episode_rewards=episode_rewards,
                       episode_values=episode_values,
                       episode_code_strings=episode_code_strings,
                       episode_actions=episode_actions,
                       episode_results=episode_results)

class ScrumMaster(Agent):
    """
    Solves reinforcement learning tasks by managing a Developer
    - directing them to write programs
    - forcing them to reprogram if the programs don't compile
    - giving them positive and negative reinforcement
    - measuring their performance
    At the moment, Scrum Master is only able to manage one developer
    """

    def __init__(self, developer, env, sprint_length, 
                 cycle_programs=True, syntax_error_reward=0):
        self.developer = developer
        # TODO: config discretization steps
        self.memory_writer = bf.TuringMemoryWriter(env.observation_space)
        self.action_sampler = bf.ActionSampler(env.action_space)
        self.cycle_programs = cycle_programs

        self.syntax_error_reward = syntax_error_reward

        self.sprint_length = sprint_length
        self.sprint_ttl = sprint_length

        self.programs_for_execution = self.make_executables()
        self.executed_programs = [self.programs_for_execution.pop()]
        self.programs_for_reflection = []
        self.rewards = []

    def init(self):
        if self.rewards:
            self.executed_programs[-1].rewards = self.rewards
            self.rewards = []
            self.programs_for_reflection.append(self.executed_programs[-1])
        self.executed_programs[-1].init()

    def input(self, inp):
        try:
            self.executed_programs[-1].input(inp)
        except bf.ProgramFinishedError:
            if self.executed_programs[-1].result != bf.Result.SUCCESS:
                self.reward(self.syntax_error_reward, force_reprogram=True)
                self.input(inp)
            else:
                raise

    def act(self):
        try:
            action = self.executed_programs[-1].act()
            return action
        except bf.ProgramFinishedError:
            if self.executed_programs[-1].result != bf.Result.SUCCESS:
                self.reward(self.syntax_error_reward, force_reprogram=True)
                return self.act()
            else:
                raise

    def value(self):
        return self.executed_programs[-1].value()

    def retrospective(self):
        if self.programs_for_reflection:
            reinforcement = programs_as_rl_episodes(self.programs_for_reflection)
            self.developer.reflect(reinforcement)
            self.programs_for_reflection = []

    def done(self):
        self.retrospective()

    def reward(self, reward, force_reprogram=False):
        self.rewards.append(reward)
        self.sprint_ttl -= 1

        if force_reprogram or self.sprint_ttl == 0:
            self.reprogram()

    def make_executables(self):
        # Get the developer to write code
        programs = self.developer.write_programs()

        # Compile it (might get syntax errors, our developer doesn't check for that)
        return [program.compile(memory_writer=self.memory_writer, 
                                action_sampler=self.action_sampler,
                                cycle=self.cycle_programs) 
                for program in programs]

    def reprogram(self):
        if len(self.programs_for_execution) == 0:
            self.programs_for_execution = self.make_executables()

        self.executed_programs[-1].rewards = self.rewards
        self.rewards = []
        self.programs_for_reflection.append(self.executed_programs[-1])
        self.executed_programs.append(self.programs_for_execution.pop())

        if len(self.programs_for_reflection) == self.developer.batch_size:
            self.retrospective()

        self.sprint_ttl = self.sprint_length