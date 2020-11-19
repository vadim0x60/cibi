import os
import shutil
import click

from pathlib import Path

@click.command()
@click.argument('exps_dir', type=str)
@click.option('-n', default=1, type=int, help='create several experiment directories, specify number')
def new_experiment(exps_dir, n):
    for idx in range(1024):
        exp_dir = os.path.join(exps_dir, str(idx))

        if os.path.isdir(exp_dir):
            continue

        os.makedirs(exp_dir)
        Path(exp_dir).joinpath('experiment.yml').touch()
        n -= 1

        if n <= 0:
            return

if __name__ == '__main__':
    new_experiment()