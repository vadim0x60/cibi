class Developer:
    name = 'developer'

    def __init__(self, language, name=None):
        self.language = language
        if name:
            self.name = name

    def write_programs(self, inspiration_branch):
        pass

    def accept_feedback(self, feedback_branch):
        pass