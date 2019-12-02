from scrum_master import ScrumMaster
from developer import Developer, hire
from lm import LanguageModel
from defaults import default_config_with_updates
from options import task_launcher
from metrics import best_reward_window

import tensorflow as tf
import numpy as np
import gym
import logging
import os

from sepsis import SepsisEnv

@task_launcher
def run_gym_test(config, task_id, logdir, summary_tasks, master, log_level, num_repetitions):
    config = default_config_with_updates(config)

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__file__)

    is_chief = (task_id == 0)
    train_dir = os.path.join(logdir, 'train')
    events_dir = '%s/events_%d' % (logdir, task_id)
    logger.info('Events directory: %s', events_dir)

    developer = Developer(config, LanguageModel)

    if not (summary_tasks and task_id < summary_tasks):
        events_dir = None

    env = SepsisEnv()

    with hire(developer, log_dir=logdir, events_dir=events_dir, is_chief=is_chief) as employed_developer:
        agent = ScrumMaster(employed_developer, env,
                            cycle_programs=True,
                            sprint_length=config.sprint_length,
                            syntax_error_reward=config.syntax_error_reward)

        for experiment_idx in range(num_repetitions):
            # Gym sets are not sets in a (data)set sense
            # These are sets in the gym "sets and reps" sense

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