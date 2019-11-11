import logging
logger = logging.getLogger(__file__)

class ActionError(Exception):
    """This agent cannot act at the moment"""
    def __init__(self, message='This agent cannot act at the moment', details=None):
        super().__init__(message)
        self.details = details

class Agent():
    def input(self, inp):
        pass

    def act(self):
        raise NotImplementedError

    def reward(self, reward):
        pass

    def attend_gym(self, env, reps=1000):
        total_reward = 0

        try:
            observation = env.reset()
            self.input(observation)
            
            for _ in range(reps):
                env.render()      
                action = self.act()

                observation, reward, done, info = env.step(action)
                total_reward += reward

                self.reward(reward)
                self.input(observation)

                if done:
                    observation = env.reset()
                env.close()
        except ActionError as e:
            logger.warn(f'Gym training finished prematurely: {e}')

        return total_reward