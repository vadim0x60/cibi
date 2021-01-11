import click
import os
import yaml
import pandas as pd

from cibi.utils import calc_hash, trusted_version

@click.command()
@click.argument('exp_dir')
def status_cmd(exp_dir):
    exp_dir = os.path.split(os.path.realpath(exp_dir))
    experiments = get_status(os.path.join(*exp_dir[:-1]), exp_dir[-1])
    index, records = zip(*experiments)
    
    try:
        # If experiment names are integers, we want them sorted as integers, not alphabetically
        index = [int(i) for i in index]
    except ValueError:
        # However, they don't have to be integers
        pass

    status_table = pd.DataFrame.from_records(records, index=index).sort_index()
    print(status_table.to_string())
    status_table.to_pickle(os.path.join(*(exp_dir + ('status.pickle',))))
    status_table.to_latex(os.path.join(*(exp_dir + ('status.tex',))))

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
        score = None

        with open(experiment_path) as experiment_f:
            experiment = yaml.safe_load(experiment_f)
            experiments.append([exp_name, experiment])

        if os.path.exists(summary_path):
            if os.path.exists(top_path):
                status = 'FINISHED'
                
                with open(top_path, 'rb') as top_f:
                    top_codebase = pd.read_pickle(top_f)
            else:
                status = 'IN PROGRESS'

            try:
                with open(summary_path) as summary_f:
                    summary = yaml.safe_load(summary_f)
                    summary = {key.replace('_', '-'): summary[key] for key in summary.keys()}

                    if not trusted_version(summary) or summary['experiment'] != calc_hash(experiment):
                        status = 'RESTART NEEDED'

                    mtr = summary['max-total-reward']
                    top_program = summary.get('top')
                    if top_program:
                        score = top_program['total_reward']
            except (yaml.YAMLError, AttributeError):
                pass

        experiment['mtr'] = mtr
        experiment['status'] = status
        experiment['score'] = score
    return experiments

if __name__ == '__main__':
    status_cmd()