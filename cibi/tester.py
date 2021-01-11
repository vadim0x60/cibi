from cibi.codebase import make_dev_codebase

class Tester():
    """
    A team member who does not create anything

    He is useful to the team, because he repeats successful programs and, 
    as a result, they get re-tested seevral times.
    Retesting is important since some programs can occasionally succeed by chance
    """

    def __init__(self, name='tester', n=3):
        self.name = name
        self.n = n

    def write_programs(self, inspiration_branch):
        dev_branch = make_dev_codebase()
        dev_branch.merge(inspiration_branch.sample(self.n, metric='quality'), force=True)
        return dev_branch

    def accept_feedback(self, feedback_branch):
        pass

    def hire(self, language, log_dir=None, events_dir=None, is_chief=True):
        self.language = language
        return self

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass