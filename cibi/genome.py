import numpy as np

def span_within_span(child, parent):
    child_start, child_end = child
    parent_start, parent_end = parent
    return child_start >= parent_start and child_end <= parent_end

structure_err = "Structure is a nested dict of ints. This ain't a dict nor an int"
blueprint_err = "Blueprint is a nested dict of numpy arrays. This ain't a dict nor a numpy array"

# TODO: Do everything via a general tree walker?

def nested_sum(structure, leaf_type, pred):
    total = 0

    for key, val in structure.items():
        t = type(val)
        if t == leaf_type:
            total += pred(val)
        elif t == dict:
            total += nested_sum(val, leaf_type, pred)
        else:
            raise ValueError(f"Expected leaf type is {leaf_type}, not {t}")

    return total

class Chromosome():
    def __init__(self, code, span=None):
        self.code = code
        if span:
            self.span = span
        else:
            self.span = (0, len(code))
        self.children = {}
        
    def add_child(self, name, span, stretch=True):
        assert self.span[0] <= span[0]
        if self.span[1] < span[1]:
            assert stretch
            self.span[1] = span[1]

        child = Chromosome(self.code, span)
        self.children[name] = child
        return child

    def structure(self, structure, stretch=True):
        start_idx = self.span[0]
        for key, length in structure.items():
            if type(length) == int:
                end_idx = start_idx + length
                self.add_child(key, (start_idx, end_idx), stretch=stretch)
                start_idx = end_idx
            elif type(length) == dict:
                child = self.add_child(key, (start_idx, start_idx), stretch=stretch)
                child.structure(length, stretch=True)
                end_idx = child.span[1]
                if end_idx > self.span[1]:
                    assert stretch
                    self.span[1] = end_idx
                start_idx = end_idx
            else:
                raise ValueError(structure_err)

    def populate(self, blueprint, stretch=True):
        start_idx = self.span[0]
        for key, val in blueprint.items():
            if type(val) == np.ndarray:
                end_idx = start_idx + len(val)
                child = self.add_child(key, (start_idx, end_idx), stretch=stretch)
                child.set(val)
                start_idx = end_idx
            elif type(val) == dict:
                child = self.add_child(key, (start_idx, start_idx), stretch=stretch)
                child.populate(val, stretch=True)
                end_idx = child.span[1]
                if end_idx > self.span[1]:
                    assert stretch
                    self.span[1] = end_idx
                start_idx = end_idx
            else:
                raise ValueError(structure_err)

    def __getitem__(self, key):
        if type(key) == str:
            return self.children[key]
        else:
            item = self
            for subkey in key:
                item = item[subkey]
            return item

    def get(self):
        start, end = self.span
        return self.code[start:end]

    def set(self, arr):
        start, end = self.span
        self.code[start:end] = arr

def make_empty_chromosome(structure):
    code = np.zeros(nested_sum(structure, int, lambda x: x))
    chromosome = Chromosome(code)
    chromosome.structure(structure)
    return chromosome
    
def make_chromosome_from_blueprint(blueprint):
    code = np.zeros(nested_sum(blueprint, np.ndarray, len))
    chromosome = Chromosome(code)
    chromosome.populate(blueprint)
    return chromosome