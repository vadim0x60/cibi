from agent import Agent
import bf

class ScrumMaster(Agent):
    """
    Solves reinforcement learning tasks by managing a Developer
    - directing them to write programs
    - forcing them to reprogram if the programs don't compile
    - giving them positive and negative reinforcement
    - measuring their performance
    At the moment, Scrum Master is only able to manage one developer
    """

    def __init__(self, developer, env, sprint_length, 
                 cycle_programs=True, syntax_error_reward=0):
        self.developer = developer
        # TODO: config discretization steps
        self.memory_writer = bf.TuringMemoryWriter(env.observation_space)
        self.action_sampler = bf.ActionSampler(env.action_space)
        self.cycle_programs = cycle_programs

        self.syntax_error_reward = syntax_error_reward

        self.sprint_length = sprint_length
        self.sprint_ttl = sprint_length

        self.programs_for_execution = self.make_executables()
        self.program = self.programs_for_execution.pop()
        self.programs_for_reflection = []
        self.rewards = []

    def input(self, inp):
        try:
            self.program.input(inp)
        except bf.ProgramFinishedError:
            if self.program.result != bf.Result.SUCCESS:
                self.reward(self.syntax_error_reward, force_reprogram=True)
                self.input(inp)
            else:
                raise

    def act(self):
        try:
            action = self.program.act()
            return action
        except bf.ProgramFinishedError:
            if self.program.result != bf.Result.SUCCESS:
                self.reward(self.syntax_error_reward, force_reprogram=True)
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

    def make_executables(self):
        # Get the developer to write code
        programs = self.developer.write_programs()

        # Compile it (might get syntax errors, our developer doesn't check for that)
        return [program.compile(memory_writer=self.memory_writer, 
                                action_sampler=self.action_sampler,
                                cycle=self.cycle_programs) 
                for program in programs]

    def reprogram(self):
        if len(self.programs_for_execution) == 0:
            self.programs_for_execution = self.make_executables()

        self.program.rewards = self.rewards
        self.rewards = []
        self.programs_for_reflection.append(self.program)
        self.program = self.programs_for_execution.pop()

        if len(self.programs_for_reflection) == self.developer.batch_size:
            self.developer.reflect(self.programs_for_reflection)