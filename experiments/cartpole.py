from cibi.scrum_master import ScrumMaster
from cibi.developer import Developer, hire
from cibi.lm import LanguageModel
from cibi.defaults import default_config_with_updates
from cibi.options import task_launcher
from cibi.metrics import best_reward_window

import tensorflow as tf
import numpy as np
import gym
import logging
import os

import tensorflow as tf

@task_launcher
def run_gym_test(config, task_id, logdir, summary_tasks, master, log_level, num_repetitions):
    config = default_config_with_updates(config)

    os.makedirs(logdir, exist_ok = True)
    parent_logger = logging.getLogger('bff')
    parent_logger.setLevel(log_level)
    parent_logger.addHandler(logging.FileHandler(f'{logdir}/log.log'))

    logger = logging.getLogger(f'bff.{__file__}')

    is_chief = (task_id == 0)
    train_dir = os.path.join(logdir, 'train')
    events_dir = '%s/events_%d' % (logdir, task_id)
    logger.info('Events directory: %s', events_dir)

    developer = Developer(config, LanguageModel)

    if not (summary_tasks and task_id < summary_tasks):
        events_dir = None

    env = gym.make('CartPole-v0')

    with hire(developer, log_dir=train_dir, events_dir=events_dir, is_chief=is_chief) as employed_developer:
        agent = ScrumMaster(employed_developer, env,
                            cycle_programs=True,
                            sprint_length=config.sprint_length,
                            syntax_error_reward=config.syntax_error_reward)

        for experiment_idx in range(num_repetitions):
            # Gym sets are not sets in a (data)set sense
            # These are sets in the gym "sets and reps" sense

            for s in range(config.gym_sets):
                agent.attend_gym(env, max_reps=config.gym_reps, render=config.render)
            
            score = best_reward_window(agent.rewards, 100)
            print(f'Best 100-episode reward: {score}')

if __name__ == '__main__':
    run_gym_test()