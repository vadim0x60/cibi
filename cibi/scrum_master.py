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
    """

    def __init__(self, developers, env, 
                 observation_discretizer, action_sampler,
                 seed_codebase = None, sprints_elapsed=0,
                 cycle_programs=True, replay_temperature=50,
                 program_file=None, quality_callback=lambda x: x):
        self.developers = developers
        self.developer_queue = itertools.cycle(developers)
        self.lead_developer = next(self.developer_queue)
        self.observation_discretizer = observation_discretizer
        self.action_sampler = action_sampler
        self.cycle_programs = cycle_programs
        self.quality_callback = quality_callback

        self.replay_temperature = replay_temperature

        self.sprints_elapsed = 0

        self.dev_branch = make_dev_codebase()
        self.feedback_branch = make_prod_codebase(deduplication=False)
        self.archive_branch = make_prod_codebase(deduplication=True, 
                                                 save_file=program_file)

        if seed_codebase and len(self.archive_branch) == 0:
            seed_codebase['quality'] = np.exp(np.array(seed_codebase['total_reward']) 
                                                    / self.replay_temperature)
            self.archive_branch.merge(seed_codebase)

        self.prod_program = None
        self.prod_rewards = []

    def __enter__(self):
        for dev in self.developers:
            dev.__enter__()
        return self

    def __exit__(self, type, value, tb):
        for dev in self.developers:
            dev.__exit__(type, value, tb)

    def finalize_episode(self):
        if self.prod_rewards:
            q = sum(self.prod_rewards)

            self.quality_callback(q)

            self.prod_program.metrics['total_reward'] = q
            self.prod_program.metrics['quality'] = exp(q / self.replay_temperature)

            self.prod_program.metadata['result'] = self.prod_program.result
            self.prod_program.metadata['author'] = self.lead_developer.name

            self.feedback_branch.commit(self.prod_program.code, 
                                        metrics=self.prod_program.metrics, 
                                        metadata=self.prod_program.metadata)
            
            self.prod_rewards = []
            self.prod_program = None

    def init(self):
        self.finalize_episode()
        if not self.prod_program:
            self.reprogram()
        self.prod_program.init()

    def input(self, inp):
        while True:
            try:
                self.prod_program.input(inp)
                return
            except bf.ProgramFinishedError:
                if self.prod_program.result != bf.Result.SUCCESS:
                    self.prod_rewards = [float('-inf')]
                    self.reprogram()
                else:
                    raise

    def act(self):
        while True:
            try:
                return self.prod_program.act()
            except bf.ProgramFinishedError:
                if self.prod_program.result != bf.Result.SUCCESS:
                    self.prod_rewards = [float('-inf')]
                    self.reprogram()
                else:
                    raise

    def retrospective(self):
        """Give the developers feedback on his code and archive said code"""
        if len(self.feedback_branch):
            self.lead_developer.accept_feedback(self.feedback_branch)
            self.archive_branch.merge(self.feedback_branch)
            self.feedback_branch.clear()

            # We are a flat team
            # Lead developer rotates every sprint
            self.lead_developer = next(self.developer_queue)

    def done(self):
        self.reprogram()

        self.archive_branch.flush()

    def reward(self, reward):
        self.prod_rewards.append(reward)

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
                                          metrics=metrics, metadata=metadata,
                                          observation_discretizer=self.observation_discretizer, 
                                          action_sampler=self.action_sampler,
                                          cycle=self.cycle_programs)
        self.sprints_elapsed += 1
        
def hire_team(developers, env, observation_discretizer, action_sampler, 
              language, log_dir, events_dir, scrum_master_args, 
              seed_codebase=None):
    employees = [dev.hire(language, log_dir, events_dir) 
                 for dev in developers]
    manager = ScrumMaster(employees, env, 
                          observation_discretizer, action_sampler,
                          seed_codebase,
                          **scrum_master_args)
    return manager