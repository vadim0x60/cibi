from cibi.roles import Developer
from cibi.codebase import make_dev_codebase
import deap.tools.mutation as deap

DEFAULT_IND_PB = 0.2

class Editor(Developer):
    def __init__(self, language, name=None, indpb=DEFAULT_IND_PB):
        super().__init__(language, name)
        self.indpb = indpb

    def mutate(self, old_code):
        raise NotImplementedError()

    def write_programs(self, inspiration_branch):
        parent_code, _, _ = inspiration_branch.sample(1, metric='quality').peek()
        
        if len(parent_code) > 1:
            # list() is important not just as a type conversion, but to prevent 
            # parent_code from modification
            child_code = ''.join(self.mut_function(self.language, list(parent_code), self.indpb))
            
        codebase = make_dev_codebase()
        codebase.commit(child_code, metadata={'author': self.name, 'parent1': parent_code})
        return codebase

class ShuffleEditor(Editor):
    name = 'shuffle_mutation'

    def mutate(self, old_code):
        return deap.mutShuffleIndexes(old_code, self.indpb)[0]

class UniformEditor(Editor):
    name = 'uniform_mutation'

    def mutate(self, old_code):
        old_code = self.language.char_to_int(old_code)
        new_code = deap.mutUniformInt(old_code, 1, len(self.language.token_space) - 1, self.indpb)[0]

        new_code = self.language.int_to_char(new_code)
        return new_code