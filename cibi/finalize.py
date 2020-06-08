import click
import os
import yaml

from cibi.codebase import make_prod_codebase
from cibi.utils import ensure_enough_test_runs
from cibi.extensions import make_gym
from cibi import bf

@click.command()
@click.argument('logdir', type=click.Path())
def finalize_training(logdir):
    with open(os.path.join(logdir, 'experiment.yml'), 'r') as f:
        config = yaml.load(f)
    discretization_config = config.get('discretization', {})
    codebase_file = os.path.join(logdir, 'programs.pickle')

    env = make_gym(config['env'])
    observation_discretizer = bf.ObservationDiscretizer(env.observation_space, 
                                                        history_length=discretization_config.get('history', 1024),
                                                        force_fluid=discretization_config.get('force-history', False))
    action_sampler = bf.ActionSampler(env.action_space)

    archive_branch = make_prod_codebase(deduplication=True, save_file=codebase_file)
    top_candidates = archive_branch.top_k('test_quality', 256)
    ensure_enough_test_runs(top_candidates, env, observation_discretizer, action_sampler)
    top_candidates.data_frame.to_pickle(os.path.join(logdir, 'top.pickle'))
    

if __name__ == '__main__':
    finalize_training()