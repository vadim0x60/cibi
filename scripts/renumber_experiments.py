import os
import shutil
import click

@click.command()
@click.argument('exp_dir', type=str)
def renumber_experiments(exp_dir):
    out_idx = 0
    for in_idx in range(1024):
        if os.path.isdir(os.path.join(exp_dir, str(in_idx))):
            if out_idx != in_idx:
                shutil.move(os.path.join(exp_dir, str(in_idx)),
                            os.path.join(exp_dir, str(out_idx)))
            out_idx += 1

if __name__ == '__main__':
    renumber_experiments()