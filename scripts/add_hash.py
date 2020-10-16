import click
import os
import yaml

from cibi.utils import calc_hash

@click.command()
@click.argument('exp_dir')
def add_hash_cmd(exp_dir):
    add_hash(exp_dir)

def add_hash(exp_dir):
    if not os.path.isdir(exp_dir):
        return

    for path in os.listdir(exp_dir):
        add_hash(os.path.join(exp_dir, path))

    summary_path = os.path.join(exp_dir, 'summary.yml')
    experiment_path = os.path.join(exp_dir, 'experiment.yml')

    if os.path.exists(summary_path) and os.path.exists(experiment_path):
        with open(summary_path, 'r') as summary_f:
            summary = yaml.safe_load(summary_f)

        with open(experiment_path, 'r') as experiment_f:
            experiment = yaml.safe_load(experiment_f)

        with open(summary_path, 'w') as summary_f:
            summary['experiment'] = calc_hash(experiment)
            yaml.dump(summary, summary_f)

if __name__ == '__main__':
    add_hash_cmd()