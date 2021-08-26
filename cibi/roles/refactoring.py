from cibi.codebase import make_dev_codebase
from cibi.roles import Developer

class Nitpicker(Developer):
    name = 'refactoring'

    def write_programs(self, inspiration_branch):
        code, _, _ = inspiration_branch.sample(1, metric='quality').peek()
        codebase = make_dev_codebase()
        metadata = {'author': self.name, 
                    'parent1': code}
        codebase.commit(self.language.prune(code), metadata=metadata)
        return codebase