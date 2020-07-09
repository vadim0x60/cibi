import click
import os
import yaml

@click.command()
@click.argument('exp_dir')
def status_cmd(exp_dir):
    exp_dir = os.path.split(os.path.realpath(exp_dir))
    status(os.path.join(*exp_dir[:-1]), exp_dir[-1])

def status(parent_dir, exp_name):
    exp_dir = os.path.join(parent_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return

    subdirs = list(os.listdir(exp_dir))
    try:
        subdirs = sorted(subdirs, key=lambda x: int(x))
    except ValueError:
        pass

    for subdir in subdirs:
        status(exp_dir, subdir)

    summary_path = os.path.join(exp_dir, 'summary.yml')
    experiment_path = os.path.join(exp_dir, 'experiment.yml')
    top_path = os.path.join(exp_dir, 'top.pickle')

    if os.path.exists(experiment_path):
        if os.path.exists(summary_path):
            with open(summary_path) as summary_f:
                mtr = yaml.safe_load(summary_f)['max_total_reward']

            if os.path.exists(top_path):
                print(f'{exp_name} - FINISHED, mtr {mtr}')
            else:
                print(f'{exp_name} - IN PROGRESS, mtr {mtr}')
        else:
            print(f'{exp_name} - NOT STARTED')

if __name__ == '__main__':
    status_cmd()