from cibi import bf
from deap.tools import crossover
import re
from numpy.random import choice

cell_actions = ''.join(bf.SHORTHAND_CELLS) + '><'

def prune(code):
    return re.sub(f'[{cell_actions}]+(?=[{bf.SHORTHAND_CELLS}])', '', code)

def mutate(code):
    pass

mate_modes = {
    '1point': crossover.cxOnePoint,
    '2point': crossover.cxTwoPoint,
    'ordered': crossover.cxOrdered
}

def mate(code1, code2, mode='random'):
    code1, code2 = list(code1), list(code2)
    if mode == 'random':
        crossover = choice(list(mate_modes.values()))
    else:
        crossover = mate_modes[mode]
    crossover(code1, code2)
    return ''.join(code1), ''.join(code2)

if __name__ == '__main__':
    print(mate('+--->>>!', '!!!'))