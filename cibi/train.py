import click
import inspect
import os
import logging
import gym

from cibi.utils import get_dir_out_of_the_way
from cibi.teams import teams
from cibi.scrum_master import hire_team
from cibi.utils import parse_config_string

def run_experiment(team_id, env_name, scrum_config, logdir, num_repetitions, num_sprints, render):
    logger = logging.getLogger('cibi')  
    logger.info(env_name)

    team = teams[team_id]
    env = gym.make(env_name)
    shortest_episode = float('inf')
    longest_episode = 0
    max_total_reward = float('-inf')

    train_dir = os.path.join(logdir, 'train')
    events_dir = os.path.join(logdir, 'events')

    with hire_team(team, env, train_dir, events_dir, scrum_config) as agent:
        while agent.sprints_elapsed < num_sprints:
            rollout = agent.attend_gym(env, max_reps=None, render=render)

            episode_length = len(rollout)
            shortest_episode = min(shortest_episode, episode_length)
            longest_episode = max(longest_episode, episode_length)
            max_total_reward = max(max_total_reward, rollout.total_reward)

            with open(os.path.join(logdir, 'summary.txt'), 'w') as f:
                summary = str({
                    'shortest_episode': shortest_episode,
                    'longest_episode': longest_episode,
                    'max_total_reward': max_total_reward
                })

                f.write(summary)
        logger.info(f'Summary: {summary}')
        
@click.command()
@click.argument('team-id', type=int)
@click.argument('env', type=str)
@click.option('--num-sprints', default=1024, type=int, help='Training length in sprints')
@click.option('--scrum-config', default='', type=str, help='Scrum configuration')
@click.option('--logdir', default='log/exp', type=str, help='Absolute path where to write results.')
@click.option('--num-repetitions', default=1, type=int, help='Number of times the same experiment will be run (globally across all workers). Each run is independent.')
@click.option('--log-level', default='INFO', type=str,  help='The threshold for what messages will be logged. One of DEBUG, INFO, WARN, ERROR, or FATAL.')
@click.option('--render', is_flag=True, help='Render the environment to monitor agents decisions')
def run_experiments(team_id, env, num_sprints, scrum_config, logdir, num_repetitions, log_level, render=True):
    scrum_config = parse_config_string(scrum_config)
    
    if num_repetitions == 1:
        experiment_dirs = [logdir]
    else:
        os.makedirs(logdir, exist_ok = True)
        experiment_dirs = [f'exp{idx}' for idx in range(num_repetitions)]
        
    for experiment_dir in experiment_dirs:
        get_dir_out_of_the_way(experiment_dir)
        os.makedirs(experiment_dir)

        if 'program_file' not in scrum_config:
            scrum_config['program_file'] = os.path.join(experiment_dir, 'programs.pickle')

        parent_logger = logging.getLogger('cibi')
        parent_logger.setLevel(log_level)
        parent_logger.addHandler(logging.FileHandler(f'{experiment_dir}/log.log'))

        run_experiment(team_id, env, scrum_config, experiment_dir, num_repetitions, num_sprints, render)

if __name__ == '__main__':
    run_experiments()