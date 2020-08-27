import click
import os
import yaml
import pandas as pd

@click.command()
@click.argument('exp_dir')
def status_cmd(exp_dir):
    exp_dir = os.path.split(os.path.realpath(exp_dir))
    experiments = get_status(os.path.join(*exp_dir[:-1]), exp_dir[-1])
    index, records = zip(*experiments)
    
    status_table = pd.DataFrame.from_records(records, index=index)
    print(status_table.to_string())
    status_table.to_pickle(os.path.join(*(exp_dir + ('status.pickle',))))

def get_status(parent_dir, exp_name):
    exp_dir = os.path.join(parent_dir, exp_name)
    if not os.path.isdir(exp_dir):
        return []

    subdirs = list(os.listdir(exp_dir))
    try:
        subdirs = sorted(subdirs, key=lambda x: int(x))
    except ValueError:
        pass

    experiments = []

    for subdir in subdirs:
        experiments.extend(get_status(exp_dir, subdir))

    summary_path = os.path.join(exp_dir, 'summary.yml')
    experiment_path = os.path.join(exp_dir, 'experiment.yml')
    top_path = os.path.join(exp_dir, 'top.pickle')

    if os.path.exists(experiment_path):
        status = 'NOT STARTED'
        mtr = None

        with open(experiment_path) as experiment_f:
            experiment = yaml.safe_load(experiment_f)
            experiments.append([exp_name, experiment])

        if os.path.exists(summary_path):
            if os.path.exists(top_path):
                status = 'FINISHED'
            else:
                status = 'IN PROGRESS'

            try:
                with open(summary_path) as summary_f:
                    mtr = yaml.safe_load(summary_f)['max_total_reward']
            except yaml.YAMLError:
                pass

        experiment['mtr'] = mtr
        experiment['status'] = status
    return experiments

if __name__ == '__main__':
    status_cmd()