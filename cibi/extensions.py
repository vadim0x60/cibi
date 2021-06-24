import gym
from heartpole import HeartPole
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

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

def gen_unity_env():
    unity_build_path = os.environ.get('UNITY_BUILD')
    unity_env = UnityEnvironment(file_name=unity_build_path)
    env = UnityToGymWrapper(unity_env)
    return env

extensions = {
    'Taxi-v3mod': TaxiEnvMultiDiscrete,
    'HeartPole-v0': HeartPole,
    'Unity-v0': gen_unity_env
}

def make_gym(gym_name):
    try:
        return extensions[gym_name]()
    except KeyError:
        return gym.make(gym_name)