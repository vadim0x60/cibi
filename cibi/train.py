import click
import inspect
import os
import logging
import gym
import time
import yaml
import traceback

from importlib_metadata import version
from evestop.generic import EVEEarlyStopping

from pathlib import Path
from importlib_resources import files

import cibi
import cibi.codebases
from cibi import bf
from cibi import bf_io
from cibi.utils import ensure_enough_test_runs, calc_hash, update_keys, trusted_version
from cibi.codebase import make_prod_codebase
from cibi.extensions import make_gym
from cibi.teams import teams
from cibi.scrum_master import hire_team
from cibi.agent import EnvError

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger('cibi')

def make_seed_codebase(seed_file, env, observation_discretizer, action_sampler):
    seed_codebase = None

    if seed_file:
        try:
            with open(seed_file, 'r') as f:
                seed_codebase = make_prod_codebase(deduplication=True)
                for code in f.readlines():
                    seed_codebase.commit(code.strip(), metadata={
                        'author': 'god'
                    }, count=0)
                logger.info('Testing programs from seed codebase')
                ensure_enough_test_runs(seed_codebase, env, observation_discretizer, action_sampler)
                logger.info(seed_codebase.to_string())
                
        except OSError as e:
            logger.error(e)

    return seed_codebase

def expand_seed_path(logdir, seed_path):
    if not seed_path:
        return None

    hardcoded_path = files(cibi.codebases).joinpath(seed_path)
    custom_path = Path(logdir).joinpath(seed_path)
    
    if hardcoded_path.exists():
        return hardcoded_path
    elif custom_path.exists():
        return custom_path
    else:
        err = f"Seed codebase {seed_path} not found"
        logger.error(err)
        raise FileNotFoundError(err)

def load_summary(logdir, config_hash):
    try:
        with open(os.path.join(logdir, 'summary.yml'), 'r') as f:
            # This is an attempt to continue a previously trusted experiment
            summary = yaml.safe_load(f)

            if 'experiment' in summary and config_hash != summary['experiment']:
                # Experiment params changed, need to start from scratch
                summary = None
            elif not trusted_version(summary):
                # Important bug has been fixed since previous run, need to start from scratch
                summary = None
    except FileNotFoundError as e:
        summary = None
      
    if summary is None:
        summary = {
            'shortest-episode': float('inf'),
            'longest-episode': 0,
            'sprints-elapsed': 0,
            'seconds-elapsed': 0,
            'max-total-reward': float('-inf'),
            'experiment': config_hash
        }

    return summary

def run_experiments(logdir, finalize_now, skip_testing):
    # PREINIT - Reading the experiment's config file
    with open(os.path.join(logdir, 'experiment.yml'), 'r') as f:
        config = yaml.safe_load(f)
        config_hash = calc_hash(config)

    logger.setLevel(config.get('log_level', 'INFO'))
    logger.addHandler(logging.FileHandler(f'{logdir}/log.log'))
    logger.info(config['env'])

    logger.info('INIT - Understanding experiment configuration and current state')
    # (the experiment can be halfway done)

    required_version = str(config['cibi-version'])
    assert version('cibi').startswith(required_version)

    render = config.get('render', False)
    discretization_config = config.get('discretization', {})

    scrum_keys = ['cycle-programs', 'syntax-error-reward', 'replay-temperature']
    scrum_config = {key.replace('-', '_'): config[key] for key in scrum_keys if key in config}

    team = teams[config['team']]
    env = make_gym(config['env'])

    max_failed_sprints = config.get('max-failed-sprints', 10)
    max_sprints = config.get('max-sprints', 1000000) * len(team)
    max_sprints_without_improvement = config.get('max-sprints-without-improvement', 1000000) * len(team)

    early_stopping = EVEEarlyStopping(mode='max', patience=max_sprints_without_improvement*len(team))

    scrum_config['program_file'] = os.path.join(logdir, 'programs.pickle')

    seed = config.get('seed')
    seed = expand_seed_path(logdir, seed)

    summary = load_summary(logdir, config_hash)
    
    summary['cibi-version'] = version('cibi')
    scrum_config['sprints_elapsed'] = summary['sprints-elapsed']
    scrum_config['quality_callback'] = early_stopping.register

    train_dir = os.path.join(logdir, 'train')
    events_dir = os.path.join(logdir, 'events')

    observation_discretizer = bf_io.ObservationDiscretizer(env.observation_space, 
                                                           history_length=discretization_config.get('history', 1024),
                                                           force_fluid=discretization_config.get('force-history', False))
    action_sampler = bf_io.ActionSampler(env.action_space)
    language = bf.make_bf_plus(config.get('allowed-commands', bf.DEFAULT_CMD_SET))

    random_agent = bf.Executable('@!', observation_discretizer, action_sampler, cycle=True, debug=False)

    try:
        bf_io.burn_in(env, random_agent, observation_discretizer, action_sampler)
    except EnvError:
        logger.error('Could not complete burn in')
        pass

    agent = None

    if not finalize_now:
        logger.info('TRAIN - Running many iterations of instant scrum')

        start_time = time.monotonic()
        start_time -= summary['seconds-elapsed']

        seed_codebase = make_seed_codebase(seed, env, observation_discretizer, action_sampler)

        failed_sprints = 0
        with hire_team(team, env, observation_discretizer, action_sampler, language,
                    train_dir, events_dir, scrum_config, seed_codebase) as agent:
            max_episode_length = config.get('max-episode-length')

            while (agent.sprints_elapsed < max_sprints and early_stopping.proceed):
                try:
                    rollout = agent.attend_gym(env, max_reps=max_episode_length, render=render)

                    episode_length = len(rollout)
                    summary['shortest-episode'] = min(summary['shortest-episode'], episode_length)
                    summary['longest-episode'] = max(summary['longest-episode'], episode_length)
                    if summary['max-total-reward'] < rollout.total_reward:
                        summary['max-total-reward'] = float(rollout.total_reward)
                        
                    summary['sprints-elapsed'] = agent.sprints_elapsed
                    summary['seconds-elapsed'] = time.monotonic() - start_time

                    with open(os.path.join(logdir, 'summary.yml'), 'w') as f:
                        yaml.dump(summary, f)

                    failed_sprints = 0
                except KeyboardInterrupt:
                    logger.info('Keyboard interrupt received. Winding down')
                    break
                except EnvError as e:
                    logger.error(traceback.format_exc())
                    failed_sprints += 1
                    if failed_sprints > max_failed_sprints:
                        logger.error('Tolerance for failed sprints exceeded')
                        raise e

    logger.info('FINALIZE - Thoroughly testing the 100 best programs found and saving the final score')

    if agent:
        archive_branch = agent.archive_branch
    else:
        archive_branch = make_prod_codebase(deduplication=True, 
                                            save_file=scrum_config['program_file'])
        assert len(archive_branch), 'Trying to finalize training with no programs written'

    top_candidates = archive_branch.top_k('total_reward', 256)
    if not skip_testing:
        ensure_enough_test_runs(top_candidates, env, observation_discretizer, action_sampler)
    top_program, top_metrics, top_metadata = top_candidates.top_k('total_reward', 1).peek()

    summary['top'] = {
        'code': str(top_program),
        'author': str(top_metadata['author']),
        'method': str(top_metadata['method']),
        'parent1': str(top_metadata['parent1']),
        'parent2': str(top_metadata['parent2']),
        'total_reward': float(top_metrics['total_reward'])
    }

    top_candidates.data_frame.to_pickle(os.path.join(logdir, 'top.pickle'))

    logger.info(f'Summary: {summary}')
    with open(os.path.join(logdir, 'summary.yml'), 'w') as f:
        yaml.dump(summary, f)

@click.command()
@click.argument('logdir', type=str)
@click.option('--finalize-now', is_flag=True)
@click.option('--skip-testing', is_flag=True)
def run_experiments_cmd(**kwargs):
    return run_experiments(**kwargs)

if __name__ == '__main__':
    run_experiments_cmd()