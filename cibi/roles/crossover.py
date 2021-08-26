import deap.tools.crossover as deap
from cibi.codebase import make_dev_codebase
from cibi.roles import Developer

DEFAULT_IND_PB = 0.2

class Mixer(Developer):
    def crossover(self, code1, code2):
        raise NotImplementedError()

    def write_programs(self, inspiration_branch):
        codebase = make_dev_codebase()

        if len(inspiration_branch) < 2:
            return codebase

        parent_code1, parent_code2 = inspiration_branch.sample(2, metric='quality')['code']
        child_code1, child_code2 = list(parent_code1), list(parent_code2)

        if len(child_code1) > 1 and len(child_code1) > 1:
            self.crossover(child_code1, child_code2)

        child_code1, child_code2 = ''.join(child_code1), ''.join(child_code2)
        metadata = {'author': self.name, 
                    'parent1': parent_code1,
                    'parent2': parent_code2}

        codebase.commit(child_code1, metadata=metadata)
        codebase.commit(child_code2, metadata=metadata)
        return codebase

class OnePointMixer(Mixer):
    def crossover(self, code1, code2):
        return deap.cxOnePoint(code1, code2)

class MessyMixer(Mixer):
    def crossover(self, code1, code2):
        return deap.cxMessyOnePoint(code1, code2)

class TwoPointMixer(Mixer):
    def crossover(self, code1, code2):
        return deap.cxTwoPoint(code1, code2)

class UniformMixer(Mixer):
    def __init__(self, language, name=None, indpb=DEFAULT_IND_PB):
        super().__init__(language, name)
        self.indpb = indpb

    def crossover(self, code1, code2):
        return deap.cxUniform(code1, code2, self.indpb)