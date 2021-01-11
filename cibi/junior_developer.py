from cibi import bf
from cibi.codebase import make_dev_codebase
from cibi.genome import make_chromosome_from_blueprint
from cibi.utils import retry

from deap.tools import crossover, mutation
import bandits
import numpy as np

from collections import OrderedDict
import re
import os
import yaml
import random

import logging
logger = logging.getLogger(f'cibi.{__file__}')

# The way a junior developer generates programs can be modeled as a multi-armed bandit problem
# The bandit has levers like
# - Check out a program from the inspiration branch and make a random change (mutation)
# - Check out 2 programs from the inspiration branch and mix them together somehow (crossover)
# - Check out a program from the inspiration branch and do minor refactoring (pruning)
# The junior developer starts out picking levers randomly and learns over time which ones work better

class MutationLever():
    def __init__(self, name, mut_function):
        self.name = name
        self.mut_function = mut_function

    def pull(self, language, inspiration_branch, indpb):
        parent_code, _, _ = inspiration_branch.sample(1, metric='quality').peek()
        
        if len(parent_code) > 1:
            # list() is important not just as a type conversion, but to prevent 
            # parent_code from modification
            child_code = ''.join(self.mut_function(language, list(parent_code), indpb))
            
        codebase = make_dev_codebase()
        codebase.commit(child_code, metadata={'method': self.name, 'parent1': parent_code})
        return codebase

class CrossoverLever():
    def __init__(self, name, crossover_function):
        self.name = name
        self.crossover_function = crossover_function

    def pull(self, language, inspiration_branch, indpb):
        codebase = make_dev_codebase()

        if len(inspiration_branch) < 2:
            return codebase

        parent_code1, parent_code2 = inspiration_branch.sample(2, metric='quality')['code']
        child_code1, child_code2 = list(parent_code1), list(parent_code2)

        if len(child_code1) > 1 and len(child_code1) > 1:
            self.crossover_function(child_code1, child_code2, indpb)

        child_code1, child_code2 = ''.join(child_code1), ''.join(child_code2)
        metadata = {'method': self.name, 
                    'parent1': parent_code1,
                    'parent2': parent_code2}

        codebase.commit(child_code1, metadata=metadata)
        codebase.commit(child_code2, metadata=metadata)
        return codebase

class RefactoringLever():
    name = 'refactoring'

    def pull(self, language, inspiration_branch, indpb):
        code, _, _ = inspiration_branch.sample(1, metric='quality').peek()
        codebase = make_dev_codebase()
        metadata = {'method': self.name, 
                    'parent1': code}
        codebase.commit(language['prune'](code), metadata=metadata)
        return codebase

def cx_with_number_arrays(language, crossover_over_numbers):
    def crossover_over_chars(c1, c2, indpb):
        c1 = language['char_to_int'](c1)
        c2 = language['char_to_int'](c2)
        res1, res2 = crossover_over_numbers(c1, c2, indpb)
        res1 = language['int_to_char'](res1)
        res2 = language['int_to_char'](res2)
        return res1, res2
    return crossover_over_chars

def mut_with_number_arrays(mutate_over_numbers):
    def mutate_over_chars(language, old_code, indpb):
        old_code = language['char_to_int'](old_code)
        new_code = mutate_over_numbers(language, old_code, indpb)

        new_code = language['int_to_char'](new_code)
        return new_code
    return mutate_over_chars

default_bandit = [
    MutationLever('shuffle_mutation', lambda language, code, indpb: mutation.mutShuffleIndexes(code, indpb)[0]),
    # We ban idx=0, because 0 means EOS and we don't want EOS popping up mid-code
    # This is very BF-specific, implementation-specific and unobvious
    # FIXME
    MutationLever('uniform_mutation', mut_with_number_arrays(
        lambda language, code, indpb: mutation.mutUniformInt(code, 1, 
                                                             len(language['alphabet']) - 1, indpb)[0])),
    CrossoverLever('1point_crossover', lambda c1, c2, indpb: crossover.cxOnePoint(c1, c2)),
    CrossoverLever('2point_crossover', lambda c1, c2, indpb: crossover.cxTwoPoint(c1, c2)),
    CrossoverLever('uniform_crossover', crossover.cxUniform),
    CrossoverLever('messy_crossover', lambda c1, c2, indpb: crossover.cxMessyOnePoint(c1, c2)),
    RefactoringLever()
]

class JuniorDeveloper():
    """
    A genetic programming-based developer
    """

    def __init__(self, indpb=0.2, name='junior', eps=0.2, bandit=default_bandit):
        self.indpb = indpb
        self.name = name
        self.policy = bandits.EpsilonGreedyPolicy(eps)

        self.action_idx = {}
        self.actions = bandit
        for idx, action in enumerate(self.actions):
            self.action_idx[action.name] = idx

        # The attributes below are required for self.policy to work
        self.k = len(bandit)
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
        action_idx = self.policy.choose(self)
        action = self.actions[action_idx]
        logger.info(f'Junior developer selected {action.name}')

        act_with_retries = retry(action.pull, attempts=10, test=lambda codebase: len(codebase) != 0)
        return act_with_retries(self.language, inspiration_branch, self.indpb)

    def accept_feedback(self, feedback_branch):
        for quality, method in zip(feedback_branch['quality'], feedback_branch['method']):
            try:
                action_idx = self.action_idx[method]
            except KeyError:
                continue

            # Remixed (Apache 2.0) from https://github.com/bgalbraith/bandits/blob/1cb5b0f7e716db13530324f04e57d1d07cfc5640/bandits/agent.py
            # BEGIN REMIX
            self.action_attempts[action_idx] += 1

            if self.gamma is None:
                g = 1 / self.action_attempts[action_idx]
            else:
                g = self.gamma
            q = self.value_estimates[action_idx]

            self.value_estimates[action_idx] += g*(quality - q)
            self.t += 1
            # END REMIX

        self.try_dump_state()

    def hire(self, language, log_dir=None, events_dir=None, is_chief=True):
        self.state_file = os.path.join(events_dir, f'{self.name}.yml') if events_dir else None
        self.try_load_state()
        self.language = language
        return self

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass