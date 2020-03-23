from cibi.agent import Agent
from cibi import bf
from cibi import utils

import numpy as np

class ScrumMaster(Agent):
    """
    Solves reinforcement learning tasks by managing Developers
    - directing them to write programs
    - forcing them to reprogram if the programs don't compile
    - giving them positive and negative reinforcement
    - measuring their performance
    At the moment, Scrum Master is only able to manage one developer
    """

    def __init__(self, developer, env, sprint_length, stretch_sprints=True,
                 cycle_programs=True, syntax_error_reward=0):
        self.developer = developer
        # TODO: config discretization steps
        self.observation_discretizer = bf.observation_discretizer(env.observation_space)
        self.action_sampler = bf.ActionSampler(env.action_space)
        self.cycle_programs = cycle_programs

        self.syntax_error_reward = syntax_error_reward

        self.sprint_length = sprint_length
        self.sprint_ttl = sprint_length
        self.stretch_sprints = stretch_sprints
        self.sprints_elapsed = 0

        self.dev_branch_programs = []

        self.feedback_branch_programs = []
        self.feedback_branch_qualities = []

        self.archive_branch_programs = []
        self.archive_branch_qualities = []

        self.prod_program = None
        self.prod_rewards = []

    def finalize_episode(self):
        if self.prod_rewards:
            self.feedback_branch_qualities.append(sum(self.prod_rewards))
            self.prod_rewards = []
            self.feedback_branch_programs.append(self.prod_program)
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

    def value(self):
        return self.prod_program.value()

    def retrospective(self):
        """Give the developer feedback on his code and archive said code"""

        assert len(self.feedback_branch_programs) == len(self.feedback_branch_qualities)

        if self.feedback_branch_programs:
            self.developer.accept_feedback(self.feedback_branch_programs, self.feedback_branch_qualities)

            self.archive_branch_programs.extend(self.feedback_branch_programs)
            self.archive_branch_qualities.extend(self.feedback_branch_qualities)

            self.feedback_branch_programs = []
            self.feedback_branch_qualities = []

    def done(self):
        self.retrospective()

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
        programs = self.developer.write_programs(self.archive_branch_programs, self.archive_branch_qualities)

        # Compile it (might get syntax errors, our developer doesn't check for that)
        programs = [program.compile(observation_discretizer=self.observation_discretizer, 
                                    action_sampler=self.action_sampler,
                                    cycle=self.cycle_programs) 
                    for program in programs]

        # Add it to the dev branch
        self.dev_branch_programs.extend(programs)

    def reprogram(self):
        if not self.dev_branch_programs:
            self.write_programs()

        self.finalize_episode()
        self.prod_program = self.dev_branch_programs.pop()

        self.sprint_ttl = self.sprint_length
        self.sprints_elapsed += 1