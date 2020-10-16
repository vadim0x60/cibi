import os
import shutil
import click
import yaml

from cibi.utils import update_keys

@click.command(help='Update experiment config files for cibi 3')
@click.argument('exp_dir', type=str)
def update_experiments_cmd(exp_dir):
    update_experiments(exp_dir)
    
def update_experiments(exp_dir):
    if not os.path.isdir(exp_dir):
        return

    experiment_path = os.path.join(exp_dir, 'experiment.yml')

    if os.path.exists(experiment_path):
        with open(experiment_path, 'r') as f:
            config = yaml.safe_load(f)

        # Team 3 was changed to a different team in cibi 2->3
        assert config['team'] != 3

        # Dashes should be used instead of underscores in cibi 3 configs
        config = update_keys(config, lambda x: x.replace('_', '-'))
        
        if 'scrum' in config:
            scrum_config = config['scrum']
            del config['scrum']
            config = {**config, **scrum_config}

        config['cibi-version'] = 3
        with open(experiment_path, 'w') as f:
            yaml.dump(config, f)

    for exp in os.listdir(exp_dir):
        update_experiments(os.path.join(exp_dir, exp))

if __name__ == '__main__':
    update_experiments_cmd()