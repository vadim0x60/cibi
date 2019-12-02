import logging
logger = logging.getLogger(__file__)
import itertools

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

    def attend_gym(self, env, max_reps=1000, render=True):
        total_reward = 0

        try:
            observation = env.reset()
            rng = range(max_reps) if max_reps else itertools.count()

            for _ in rng:
                self.input(observation)

                if render:
                    try:
                        env.render()  
                    except NotImplementedError:
                        render = False
                    
                action = self.act()

                observation, reward, done, info = env.step(action)

                logger.info(f'Got reward {reward}')
                total_reward += reward
                self.reward(reward)

                if done:
                    break
                
        except ActionError as e:
            logger.warn(f'Gym training finished prematurely: {e}')

        env.close()

        return total_reward