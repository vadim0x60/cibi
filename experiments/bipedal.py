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
def run_gym_test(config, task_id, logdir, summary_tasks, master, num_repetitions):
    is_chief = (task_id == 0)
    logger = logging.getLogger('cibi')
    logger.info('BipedalWalker-v2')

    env = gym.make('BipedalWalker-v2')

    train_dir = os.path.join(logdir, 'train')
    events_dir = '%s/events_%d' % (logdir, task_id)
    rollouts_dir = os.path.join(logdir, 'rollouts')
    os.makedirs(rollouts_dir)

    if not (summary_tasks and task_id < summary_tasks):
        events_dir = None
    
    total_rewards = []
    episode_lengths = []
    developer = Developer(config, LanguageModel)
    with hire(developer, log_dir=train_dir, events_dir=events_dir, is_chief=is_chief) as employed_developer:
        agent = ScrumMaster(employed_developer, env,
                            cycle_programs=True,
                            sprint_length=100,
                            stretch_sprints=False,
                            syntax_error_reward=-100)

        while agent.sprints_elapsed < config.sprints:
            rollout = agent.attend_gym(env, max_reps=None, render=config.render)
            total_rewards.append(rollout.total_reward)
            episode_lengths.append(len(rollout))

            with open(os.path.join(rollouts_dir, f'{agent.sprints_elapsed}sprints_in.dill'), 'wb') as f:
                dill.dump(rollout, f)

            with open(os.path.join(logdir, 'programs.txt'), 'w') as f:
                f.writelines(p.code + '\n' for p in agent.executed_programs)

            with open(os.path.join(logdir, 'summary.txt'), 'w') as f:
                summary = str({
                    'episode_lengths': episode_lengths,
                    'sprint_length': agent.sprint_length,
                    'shortest_episode': min(episode_lengths),
                    'total_rewards': total_rewards,
                    'max_total_reward': max(total_rewards)
                })

                f.write(summary)
        logger.info(f'Summary: {summary}')

if __name__ == '__main__':
    run_gym_test()