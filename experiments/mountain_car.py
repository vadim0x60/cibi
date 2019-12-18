from cibi.scrum_master import ScrumMaster
from cibi.developer import Developer, hire
from cibi.lm import LanguageModel
from cibi.defaults import default_config_with_updates
from cibi.options import task_launcher
from cibi.metrics import best_reward_window
from cibi.utils import get_dir_out_of_the_way

import tensorflow as tf
import numpy as np
import gym
import logging
import os
import dill

import tensorflow as tf

@task_launcher
def run_gym_test(config, task_id, logdir, summary_tasks, master, log_level, num_repetitions):
    config = default_config_with_updates(config)

    os.makedirs(logdir, exist_ok = True)
    parent_logger = logging.getLogger('bff')
    parent_logger.setLevel(log_level)
    parent_logger.addHandler(logging.FileHandler(f'{logdir}/log.log'))

    is_chief = (task_id == 0)

    env = gym.make('MountainCar-v0')

    for experiment_idx in range(num_repetitions):
        experiment_dir = os.path.join(logdir, f'exp{experiment_idx}')
        try:
            os.makedirs(experiment_dir)
        except FileExistsError:
            get_dir_out_of_the_way(experiment_dir)
            os.makedirs(experiment_dir)

        train_dir = os.path.join(experiment_dir, 'train')
        events_dir = '%s/events_%d' % (logdir, task_id)

        if not (summary_tasks and task_id < summary_tasks):
            events_dir = None
        
        rollouts = []

        developer = Developer(config, LanguageModel)
        with hire(developer, log_dir=train_dir, events_dir=events_dir, is_chief=is_chief) as employed_developer:
            agent = ScrumMaster(employed_developer, env,
                                cycle_programs=True,
                                sprint_length=config.sprint_length,
                                syntax_error_reward=config.syntax_error_reward)

            for s in range(config.gym_sets):
                rollouts.append(agent.attend_gym(env, max_reps=config.gym_reps, render=config.render))

                with open(os.path.join(experiment_dir, 'rollouts.dill'), 'wb') as f:
                    dill.dump(rollouts, f)

                with open(os.path.join(experiment_dir, 'programs.txt'), 'w') as f:
                    f.writelines(p.code + '\n' for p in agent.executed_programs)

                with open(os.path.join(experiment_dir, 'summary.txt'), 'w') as f:
                    episode_lengths = [len(rollout) for rollout in rollouts]
                    f.write(str({
                        'episode_lengths': episode_lengths,
                        'sprint_length': agent.sprint_length,
                        'shortest_episode': len(episode_lengths)
                    }))

if __name__ == '__main__':
    run_gym_test()