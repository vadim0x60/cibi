import gym_sepsis.envs.sepsis_env as e
import tensorflow as tf

class EnvWithGraph():
    def __init__(self, env, graph):
        self.env = env
        self.graph = graph
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range

    def step(self, *args):
        with self.graph.as_default():
            return self.env.step(*args)

    def reset(self, *args):
        with self.graph.as_default():
            return self.env.reset(*args)

    def render(self, *args):
        with self.graph.as_default():
            return self.env.render(*args)

    def close(self, *args):
        with self.graph.as_default():
            return self.env.close(*args)

    def seed(self, *args):
        with self.graph.as_default():
            return self.env.seed(*args)

def SepsisEnv():
    graph = tf.Graph()
    with graph.as_default():
        env = e.SepsisEnv()
    return EnvWithGraph(env, graph)