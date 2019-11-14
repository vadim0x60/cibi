from scrum_master import ScrumMaster
from developer import Developer, FullStackDeveloper
from lm import LanguageModel
from defaults import default_config_with_updates
from options import task_launcher
from metrics import best_reward_window

import tensorflow as tf
import gym
import logging
import os


@task_launcher
def run_gym_test(config, task_id, logdir, summary_tasks, master, log_level):
    config = default_config_with_updates(config)

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__file__)

    is_chief = (task_id == 0)
    train_dir = os.path.join(logdir, 'train')
    best_model_checkpoint = os.path.join(train_dir, 'best.ckpt')
    events_dir = '%s/events_%d' % (logdir, task_id)
    logger.info('Events directory: %s', events_dir)

    developer = Developer(config, LanguageModel, 
                          cycle_program=True, 
                          best_checkpoint_file=best_model_checkpoint)

    if summary_tasks and task_id < summary_tasks:
        summary_writer = tf.summary.FileWriter(events_dir)
    else:
        summary_writer = None

    def init_fn(unused_sess):
        logger.info('No checkpoint found. Initialized global params.')

    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=train_dir,
                             saver=developer.saver,
                             summary_op=None,
                             init_op=developer.global_init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=developer.ready_op,
                             ready_for_local_init_op=None,
                             global_step=developer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    with sv.managed_session(master) as session:
        developer.initialize(session)

        env = gym.make('NChain-v0')
        if type(env.action_space) == gym.spaces.Discrete:
            coerce_action = lambda action: action % env.action_space.n
        else:
            coerce_action = lambda x: x
        agent = ScrumMaster(FullStackDeveloper(developer, session),
                            sprint_length=config.sprint_length, 
                            coerce_action=coerce_action)
        env.reset()

        # Gym sets are not sets in a (data)set sense
        # These are sets in the gym "sets and reps" sense
        for s in range(config.gym_sets):
            agent.attend_gym(env, reps=config.gym_reps)

        metric = best_reward_window(agent.rewards, window_size=100)
        logger.info(f'Best 100-reps reward: {metric}')
        env.close()

    logging.basicConfig()

if __name__ == '__main__':
    run_gym_test()