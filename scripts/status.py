import click
import os

@click.command()
@click.argument('exp_dir')
def status_cmd(exp_dir):
    status(exp_dir)

def status(exp_dir):
    if not os.path.isdir(exp_dir):
        return

    for path in os.listdir(exp_dir):
        status(os.path.join(exp_dir, path))

    summary_path = os.path.join(exp_dir, 'summary.yml')
    experiment_path = os.path.join(exp_dir, 'experiment.yml')
    top_path = os.path.join(exp_dir, 'top.pickle')
    exp_name = os.path.split(exp_dir)[-1]

    if os.path.exists(experiment_path):
        if os.path.exists(summary_path):
            if os.path.exists(top_path):
                print(f'{exp_name} - FINISHED')
            else:
                print(f'{exp_name} - IN PROGRESS')
        else:
            print(f'{exp_name} - NOT STARTED')

if __name__ == '__main__':
    status_cmd()