from cibi.agent import Agent
from cibi import bf
from cibi import utils
from cibi.codebase import make_dev_codebase, make_prod_codebase

from math import exp, log
import numpy as np
import itertools

class ScrumMaster(Agent):
    """
    Solves reinforcement learning tasks by managing Developers
    - directing them to write programs
    - forcing them to reprogram if the programs don't compile
    - giving them positive and negative reinforcement
    - measuring their performance
    At the moment, Scrum Master is only able to manage one developer
    """

    def __init__(self, developers, env, sprint_length, stretch_sprints=True,
                 cycle_programs=True, syntax_error_reward=0, replay_temperature=0):
        self.developers = itertools.cycle(developers)
        self.lead_developer = next(self.developers)
        # TODO: config discretization steps
        self.observation_discretizer = bf.observation_discretizer(env.observation_space)
        self.action_sampler = bf.ActionSampler(env.action_space)
        self.cycle_programs = cycle_programs

        self.syntax_error_reward = syntax_error_reward
        self.replay_temperature = replay_temperature

        self.sprint_length = sprint_length
        self.sprint_ttl = sprint_length
        self.stretch_sprints = stretch_sprints
        self.sprints_elapsed = 0

        self.dev_branch = make_dev_codebase()
        self.feedback_branch = make_prod_codebase()
        self.archive_branch = make_prod_codebase()

        self.prod_program = None
        self.prod_rewards = []

    def finalize_episode(self):
        if self.prod_rewards:
            q = sum(self.prod_rewards)

            metrics = {
                'test_quality': q,
                'replay_weight': exp(q / self.replay_temperature),
                'log_prob': self.prod_program.log_prob
            }

            metadata = {
                'result': self.prod_program.result
            }

            self.feedback_branch.commit(self.prod_program.code, 
                                        metrics=metrics, metadata=metadata)
            
            self.prod_rewards = []
            self.prod_program = None

    def init(self):
        self.finalize_episode()
        if not self.prod_program:
            self.reprogram()
        self.prod_program.init()

    def input(self, inp):
        try:
            self.prod_program.input(inp)
        except bf.ProgramFinishedError:
            if self.prod_program.result != bf.Result.SUCCESS:
                self.prod_rewards = [self.syntax_error_reward]
                self.reprogram()
                self.input(inp)
            else:
                raise

    def act(self):
        try:
            action = self.prod_program.act()
            return action
        except bf.ProgramFinishedError:
            if self.prod_program.result != bf.Result.SUCCESS:
                self.prod_rewards = [self.syntax_error_reward]
                self.reprogram()
                return self.act()
            else:
                raise

    def retrospective(self):
        """Give the developers feedback on his code and archive said code"""
        if len(self.feedback_branch):
            self.lead_developer.accept_feedback(self.feedback_branch)
            self.archive_branch.merge(self.feedback_branch)
            self.feedback_branch.clear()

    def done(self):
        if self.sprint_ttl <= 0:
            self.reprogram()

    def reward(self, reward):
        self.prod_rewards.append(reward)
        self.sprint_ttl -= 1
            
        if not self.stretch_sprints and self.sprint_ttl == 0:
            self.reprogram()

    def write_programs(self):
        # If we have something to discuss, discuss before writing new code
        self.retrospective()

        # Get the developer to write code
        programs = self.lead_developer.write_programs(self.archive_branch)

        # Add it to the dev branch
        self.dev_branch.merge(programs)

    def reprogram(self):
        self.finalize_episode()

        if not len(self.dev_branch):
            self.write_programs()

        # Check out a program from the dev branch
        code, metrics, metadata = self.dev_branch.pop()

        # Compile it (might get syntax errors, our developer doesn't check for that)
        self.prod_program = bf.Executable(code,
                                          log_prob=metrics['log_prob'],
                                          observation_discretizer=self.observation_discretizer, 
                                          action_sampler=self.action_sampler,
                                          cycle=self.cycle_programs)
        self.sprint_ttl = self.sprint_length
        self.sprints_elapsed += 1
        
        # We are a flat team
        # Lead developer rotates every sprint
        self.lead_developer = next(self.developers)