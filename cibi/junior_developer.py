from cibi import bf
from cibi.codebase import make_dev_codebase
from cibi.genome import make_chromosome_from_blueprint

from deap.tools import crossover, mutation
import re
import numpy as np
import random

import logging
logger = logging.getLogger(f'cibi.{__file__}')

def select(elements, weights, k=1):
    weights /= np.sum(weights)
    chosen = np.random.choice(len(elements), size=k, replace=False, p=weights)
    return [elements[c] for c in chosen]

def mut_with_number_arrays(mutate_over_numbers):
    def mutate_over_chars(language, old_code, indpb):
        old_code = language['char_to_int'](old_code)
        new_code = mutate_over_numbers(language, old_code, indpb)

        new_code = language['int_to_char'](new_code)
        return new_code
    return mutate_over_chars

mutation_modes = {
    'shuffle': lambda language, code, indpb: mutation.mutShuffleIndexes(code, indpb)[0],
    # We ban idx=0, because 0 means EOS and we don't want EOS popping up mid-code
    # This is very BF-specific, implementation-specific and unobvious
    # FIXME
    'uniform': mut_with_number_arrays(
        lambda language, code, indpb: mutation.mutUniformInt(code, 1, 
                                                             len(language['alphabet']) - 1, indpb)[0])
}

def cx_with_number_arrays(language, crossover_over_numbers):
    def crossover_over_chars(c1, c2, indpb):
        c1 = language['char_to_int'](c1)
        c2 = language['char_to_int'](c2)
        res1, res2 = crossover_over_numbers(c1, c2, indpb)
        res1 = language['int_to_char'](res1)
        res2 = language['int_to_char'](res2)
        return res1, res2
    return crossover_over_chars

mating_modes = {
    '1point': lambda c1, c2, indpb: crossover.cxOnePoint(c1, c2),
    '2point': lambda c1, c2, indpb: crossover.cxTwoPoint(c1, c2),
    'uniform': crossover.cxUniform,
    'messy': lambda c1, c2, indpb: crossover.cxMessyOnePoint(c1, c2)
}

# We treat all functions from DEAP mutation and crossover
# as if they don't modify the programs in-place, but they do
# Doesn't matter here, since it happens to throwaway variables
# but beware!

class JuniorDeveloper():
    def __init__(self, strategy=None, name='junior'):
        self.available_actions = {
            'prune': self.prune,
            'mutate': self.mutate,
            'mate': self.mate
        }

        default_strategy = {
            # All options equal. Sounds like a good default
            'action_distribution': np.ones(len(self.available_actions)),
            'mutation_modes_distribution': np.ones(len(mutation_modes)),
            'mating_modes_distribution': np.ones(len(mating_modes)),
            'indpb': np.array([0.2])
        }

        if strategy is None:
            strategy = make_chromosome_from_blueprint(default_strategy)
        self.strategy = strategy
        self.name = name

    def mutate(self, inspiration_branch):  
        code, _, _ = inspiration_branch.sample(1, metric='test_quality').peek()
        
        if len(code) > 1:
            mutation_name, mutation = select(list(mutation_modes.items()), 
                                             weights=self.strategy['mutation_modes_distribution'].get())[0]
            logger.info(f'{mutation_name} mutation of {code}')
            code = ''.join(mutation(self.language, list(code), self.strategy['indpb'].get()))
            
        codebase = make_dev_codebase()
        codebase.commit(code)
        return codebase

    def mate(self, inspiration_branch):
        code1, code2 = inspiration_branch.sample(2, metric='test_quality')['code']
        code1, code2 = list(code1), list(code2)

        if len(code1) > 1 and len(code2) > 1:
            crossover_name, crossover = select(list(mating_modes.items()), 
                                               weights=self.strategy['mating_modes_distribution'].get())[0]
            logger.info(f'{crossover_name} crossover between {code1} and {code2}')
            crossover(code1, code2, self.strategy['indpb'].get())

        codebase = make_dev_codebase()
        codebase.commit(''.join(code1))
        codebase.commit(''.join(code2))
        return codebase

    def prune(self, inspiration_branch):
        code, _, _ = inspiration_branch.sample(1, metric='test_quality').peek()
        logger.info(f'pruning {code}')
        codebase = make_dev_codebase()
        codebase.commit(self.language['prune'](code))
        return codebase

    def write_programs(self, inspiration_branch):
        action_distribution = self.strategy['action_distribution'].get()
        action_name, act = select(list(self.available_actions.items()), 
                                  weights=action_distribution)[0]
        logger.info(f'Junior developer decided to {action_name}')
        return act(inspiration_branch)

    def accept_feedback(self, feedback_branch):
        logger.info('If they were good at processing feedback, they wouldn\'t be a junior developer')

    def hire(self, language, log_dir=None, events_dir=None, is_chief=True):
        self.language = language
        return self

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass