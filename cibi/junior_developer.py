from cibi import bf
from deap.tools import crossover, mutation
import re
import numpy as np
import random

import logging
logger = logging.getLogger(f'cibi.{__file__}')

cell_actions = ''.join(bf.SHORTHAND_CELLS) + '><'

def select(elements, weights, k=1):
    weights /= np.sum(weights)
    chosen = np.random.choice(len(elements), size=k, replace=False, p=weights)
    return [elements[c] for c in chosen]

def prune(program_pool, program_qualities, strategy):
    old_code = select(program_pool, program_qualities)[0].code
    logger.info(f'pruning {old_code}')
    new_code = re.sub(f'[{cell_actions}]+(?=[{bf.SHORTHAND_CELLS}])', '', old_code)
    if new_code != old_code:
        program_pool.append(bf.Program(new_code))

def mut_with_number_arrays(mutate_over_numbers):
    def mutate_over_chars(old_code, indpb):
        old_code = bf.bf_char_to_int(old_code)
        new_code = mutate_over_numbers(old_code, indpb)

        new_code = bf.bf_int_to_char(new_code)
        return new_code
    return mutate_over_chars

mutation_modes = {
    'shuffle': lambda code, indpb: mutation.mutShuffleIndexes(code, indpb)[0],
    # We ban idx=0, because 0 means EOS and we don't want EOS popping up mid-code
    # This is very BF-specific, implementation-specific and unobvious
    # FIXME
    'uniform': mut_with_number_arrays(lambda code, indpb: mutation.mutUniformInt(code, 1, len(bf.BF_INT_TO_CHAR), indpb)[0])
}

def mutate(program_pool, program_qualities, strategy):  
    old_code = select(program_pool, program_qualities)[0].code
    mutation_name, mutation = select(list(mutation_modes.items()), weights=strategy['mutation_modes_distribution'])[0]
    logger.info(f'{mutation_name} mutation of {old_code}')
    new_code = ''.join(mutation(list(old_code), strategy['indpb']))
    if old_code != new_code:
        program_pool.append(bf.Program(new_code))

def cx_with_number_arrays(crossover_over_numbers):
    def crossover_over_chars(c1, c2, indpb):
        c1 = bf.bf_char_to_int(c1)
        c2 = bf.bf_char_to_int(c2)
        res1, res2 = crossover_over_numbers(c1, c2, indpb)
        res1 = bf.bf_int_to_char(res1)
        res2 = bf.bf_int_to_char(res2)
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

def mate(program_pool, program_qualities, strategy):
    program1, program2 = select(program_pool, program_qualities, k=2)
    code1, code2 = list(program1.code), list(program2.code)
    crossover_name, crossover = select(list(mating_modes.items()), weights=strategy['mating_modes_distribution'])[0]
    logger.info(f'{crossover_name} crossover between {code1} and {code2}')
    crossover(code1, code2, strategy['indpb'])
    program_pool.append(bf.Program(''.join(code1)))
    program_pool.append(bf.Program(''.join(code2)))

available_actions = {
    'prune': prune,
    'mutate': mutate,
    'mate': mate
}

strategy_genome = {
    'action_distribution': len(available_actions),
    'mutation_modes_distribution': len(mutation_modes),
    'mating_modes_distribution': len(mating_modes),
    'indpb': 1
}

default_strategy = {
    # All options equal. Sounds like a good default
    'action_distribution': np.ones(len(available_actions)),
    'mutation_modes_distribution': np.ones(len(mutation_modes)),
    'mating_modes_distribution': np.ones(len(mating_modes)),
    'indpb': 0.2
}

def parse_strategy_vector(strategy_vector):
    strategy = {}
    for param_name, param_size in strategy_genome.items():
        strategy[param_name] = strategy_vector[:param_size]
        strategy_vector = strategy_vector[param_size:]
    return strategy

class JuniorDeveloper():
    def __init__(self, strategy_vector=None):
        if strategy_vector:
            self.strategy_vector = strategy_vector
            self.strategy = parse_strategy_vector(strategy_vector)
        else:
            self.strategy = default_strategy

    def develop(self, program_pool, program_qualities):
        action_distribution = self.strategy['action_distribution']
        action_name, act = select(list(available_actions.items()), weights=action_distribution)[0]
        logger.info(f'Junior developer decided to {action_name}')
        act(program_pool, program_qualities, self.strategy)

if __name__ == '__main__':
    import logging.handlers
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    dev = JuniorDeveloper()
    program_pool = [
        "+.>4ae.",
        "+.e.4e",
        "2.c-[4c,][4][]+>e3-",
        "c1,>[d,+..2]e<"
    ]
    program_pool = [bf.Program(p) for p in program_pool]
    dev.develop(program_pool, np.ones(4))
    print([p.code for p in program_pool])