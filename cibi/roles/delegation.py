from cibi.codebase import make_dev_codebase
from cibi.utils import retry
from cibi.roles import Developer

import bandits
import numpy as np
import os
import yaml

import logging
logger = logging.getLogger(f'cibi.{__file__}')

class LeadDeveloper(Developer):
    """
    A developer that writes code by delegating to other developers
    peons write the code and trainees learn from peons' mistakes
    the set of peons can (an should) be a subset of the set of trainees
    """

    def __init__(self, peons, trainees, indpb=0.2, name='junior', eps=0.2):
        self.indpb = indpb
        self.name = name

        self.peon_idx = {}
        self.peons = peons
        self.trainees = trainees

        for idx, peon in enumerate(self.peons):
            self.peon_idx[peon.name] = idx

        # The task of choosing the correct assistant for the job is a k-armed bandit
        self.policy = bandits.EpsilonGreedyPolicy(eps)
        # The attributes below are required for self.policy to work
        self.k = len(self.peons)
        self.value_estimates = np.zeros(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.gamma = None

    def try_dump_state(self):
        if not self.state_file:
            return

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        with open(self.state_file, 'w') as f:
            state = {
                'k': self.k,
                'value_estimates': self.value_estimates.tolist(), 
                'action_attempts': self.action_attempts.tolist(), 
                't': self.t, 
                'gamma': self.gamma
            }
            yaml.dump(state, f)

    def try_load_state(self):
        if not self.state_file:
            return

        try:
            with open(self.state_file, 'r') as f:
                state = yaml.safe_load(f)

                self.k = state['k']
                self.value_estimates = np.array(state['value_estimates'])
                self.action_attempts = np.array(state['action_attempts'])
                self.t = state['t']
                self.gamma = state['gamma']

        except OSError:
            pass

    def write_programs(self, inspiration_branch):
        peon_idx = self.policy.choose(self)
        peon = self.peon[peon_idx]
        logger.info(f'{peon.name}: I choose you!')

        write_with_retries = retry(peon.write_programs, attempts=10, test=lambda codebase: len(codebase) != 0)
        return write_with_retries(self.language, inspiration_branch, self.indpb)

    def update_bandit_policy(self, quality, author):
        try:
            peon_idx = self.peon_idx[author]
        except KeyError:
            # The author may be, for example, god (a human developer whose program we're using for inspiration)
            return

        # Remixed (Apache 2.0) from https://github.com/bgalbraith/bandits/blob/1cb5b0f7e716db13530324f04e57d1d07cfc5640/bandits/agent.py
        # BEGIN REMIX
        self.action_attempts[peon_idx] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[peon_idx]
        else:
            g = self.gamma
        q = self.value_estimates[peon_idx]

        self.value_estimates[peon_idx] += g*(quality - q)
        self.t += 1
        # END REMIX

    def accept_feedback(self, feedback_branch):
        # Train thyself
        for quality, author in zip(feedback_branch['quality'], feedback_branch['author']):
            self.update_bandit_policy(quality, author)

        self.try_dump_state()

        # Train the trainees
        for trainee in self.trainees:
            trainee.accept_feedback(feedback_branch)

    def hire(self, language, log_dir=None, events_dir=None, is_chief=True):
        self.state_file = os.path.join(events_dir, f'{self.name}.yml') if events_dir else None
        self.try_load_state()
        self.language = language
        return self

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass