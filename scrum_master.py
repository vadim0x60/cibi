from agent import Agent
from bf import ProgramFinishedError, Result

class ScrumMaster(Agent):
    """
    Solves reinforcement learning tasks by managing a Developer
    - directing them to write programs
    - forcing them to reprogram if the programs don't compile
    - giving them positive and negative reinforcement
    - measuring their performance
    At the moment, Scrum Master is only able to manage one developer
    """

    def __init__(self, developer, sprint_length, coerce_action=lambda x: x):
        self.developer = developer
        self.coerce_action = coerce_action

        self.sprint_length = sprint_length
        self.sprint_ttl = sprint_length

        self.programs_for_execution = self.developer.write_programs()
        self.program = self.programs_for_execution.pop()
        self.programs_for_reflection = []
        self.rewards = []

    def input(self, inp):
        try:
            self.program.input(inp)
        except ProgramFinishedError:
            if self.program.result != Result.SUCCESS:
                self.reward(0, force_reprogram=True)
                self.input(inp)
            else:
                raise

    def act(self):
        try:
            action = self.program.act()
            return self.coerce_action(action)
        except ProgramFinishedError:
            if self.program.result != Result.SUCCESS:
                self.reward(0, force_reprogram=True)
                return self.act()
            else:
                raise

    def reward(self, reward, force_reprogram=False):
        self.rewards.append(reward)
        self.sprint_ttl -= 1

        if force_reprogram or self.sprint_ttl == 0:
            self.reprogram()
        else:
            self.sprint_ttl = self.sprint_length

    def reprogram(self):
        if len(self.programs_for_execution) == 0:
            self.programs_for_execution = self.developer.write_programs()

        self.program.rewards = self.rewards
        self.rewards = []
        self.programs_for_reflection.append(self.program)
        self.program = self.programs_for_execution.pop()

        if len(self.programs_for_reflection) == self.developer.batch_size:
            self.developer.reflect(self.programs_for_reflection)