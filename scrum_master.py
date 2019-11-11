from agent import Agent

class ScrumMaster(Agent):
    """
    Solves reinforcement learning tasks by asking the developer to 
    write programs to solve them and running the programs
    At the moment, Scrum master is only able to manage one developer
    """

    def __init__(self, developer, sprint_length=100):
        self.developer = developer

        self.sprint_length = sprint_length
        self.sprint_ttl = sprint_length

        self.programs_for_execution = self.developer.write_programs()
        self.program = self.programs_for_execution.pop()
        self.programs_for_reflection = []
        self.rewards = []

    def input(self, inp):
        self.program.input(inp)

    def act(self):
        return self.program.act()

    def reward(self, reward):
        self.rewards.append(reward)
        self.sprint_ttl -= 1

        if self.sprint_ttl == 0:
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