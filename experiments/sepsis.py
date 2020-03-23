from cibi.scrum_master import ScrumMaster
from cibi.senior_developer import SeniorDeveloper, hire
from cibi.lm import LanguageModel
from cibi.defaults import default_config_with_updates
from cibi.launcher import task_launcher

import tensorflow as tf
import numpy as np
import gym
import logging
import os

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


@task_launcher
def run_gym_test(config, task_id, logdir, summary_tasks, master, num_repetitions):
    logger = logging.getLogger(f'cibi.{__file__}')

    is_chief = (task_id == 0)
    train_dir = os.path.join(logdir, 'train')
    events_dir = '%s/events_%d' % (logdir, task_id)
    logger.info('Events directory: %s', events_dir)

    developer = SeniorDeveloper(config, LanguageModel)

    if not (summary_tasks and task_id < summary_tasks):
        events_dir = None

    env = SepsisEnv()

    with hire(developer, log_dir=train_dir, events_dir=events_dir, is_chief=is_chief) as employed_developer:
        agent = ScrumMaster(employed_developer, env,
                            cycle_programs=True,
                            sprint_length=config.sprint_length,
                            syntax_error_reward=config.syntax_error_reward)

        logger.info(f'Train cohort of patients (it is OK to kill them for exploration)')
        for s in range(config.gym_sets):
            agent.attend_gym(env, max_reps=None, render=config.render)

        logger.info(f'Test cohort of patients (hopefully they get out alive)')
        survival_rate = 0
        steps_taken = []
        for s in range(config.gym_sets):
            agent.attend_gym(env, max_reps=None, render=config.render)
            survival_rate += agent.rewards[-1] > 0
            steps_taken.append(len(agent.rewards))

        survival_rate /= config.gym_sets
        
        logger.info(f'Survival_rate: {survival_rate}')
        logger.info(f'Steps taken: {np.mean(steps_taken)}')

if __name__ == '__main__':
    run_gym_test()