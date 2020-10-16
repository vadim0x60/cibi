import os
import shutil
import click

from cibi.utils import get_project_dir

def parse_ranges(ranges):
    for r in ranges.split(','):
        limits = r.split('-')

        if len(limits) == 1:
            yield int(limits[0])
        else:
            for idx in range(int(limits[0]), int(limits[-1])):
                yield idx

@click.command()
@click.argument('exp_dir', type=str)
@click.argument('ids', type=str)
def duplicate_experiments(exp_dir, ids):
    max_experiment_id = max(*(int(idx) for idx in os.listdir(exp_dir)))
    for idx in parse_ranges(ids):
        max_experiment_id += 1
        shutil.copytree(os.path.join(exp_dir, str(idx)),
                        os.path.join(exp_dir, str(max_experiment_id)))

if __name__ == '__main__':
    duplicate_experiments()