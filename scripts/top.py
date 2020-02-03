from sortedcontainers import SortedDict 
import re
import click

def get_top_programs(logfile, n=None):
    sorted_programs = SortedDict()

    with open(logfile) as logf:
        for line in logf.readlines():
            match = re.fullmatch('.*NPE: ([0-9]+).*Tot R: ([\.\-0-9]+)\..*Program: ([><\^\+-\[\]\.,!01234abcde]*)\n', line)
            if match:
                idx, reward, program = match.groups()
                idx = int(idx)
                reward = float(reward)
                sorted_programs[reward] = (idx, program)

    if n:
        for i in range(n):
            if not sorted_programs:
                break

            reward, (idx, program) = sorted_programs.popitem()
            yield reward, idx, program
    else:
        while sorted_programs:
            reward, (idx, program) = sorted_programs.popitem()
            yield reward, idx, program

@click.command()
@click.argument('logfile', type=str)
@click.option('-n', default=1000, type=int)
def top(logfile, n):
    with open('top.txt', 'w') as outf:
        outf.writelines([f'{reward} {idx} {program}\n' for reward, idx, program in get_top_programs(logfile, n)])
            
if __name__ == "__main__":
    top()
