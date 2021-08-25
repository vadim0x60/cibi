import gym
from heartpole import HeartPole
import auto_als

import os
class TaxiEnvMultiDiscrete(gym.Env):
    def __init__(self):
        self.openai_gym = gym.make('Taxi-v3')
        
        self.reward_range = self.openai_gym.reward_range
        self.action_space = self.openai_gym.action_space
        self.observation_space = gym.spaces.MultiDiscrete([5, 5, 5, 4])

    def step(self, *args, **kwargs):
        observation, reward, done, info = self.openai_gym.step(*args, **kwargs)
        observation = list(self.openai_gym.decode(observation))
        return observation, reward, done, info

    def reset(self, *args, **kwargs):
        observation = self.openai_gym.reset(*args, **kwargs)
        observation = list(self.openai_gym.decode(observation))
        return observation

    def render(self, *args, **kwargs):
        return self.openai_gym.render(*args, **kwargs)

extensions = {
    'Taxi-v3mod': TaxiEnvMultiDiscrete,
    'HeartPole-v0': HeartPole
}

def make_gym(gym_name):
    try:
        return extensions[gym_name]()
    except KeyError:
        return gym.make(gym_name)