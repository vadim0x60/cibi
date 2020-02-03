import click
from cibi import bf
from sortedcontainers import SortedDict
import gym
import time

@click.command()
@click.argument('env_name', required=True, type=str)
@click.argument('input_code', required=False, type=str)
@click.option('--avg', help='Average results over multiple runs', type=int, default=1)
@click.option('--input-file', '-i', help='Load the programs from a specified file')
@click.option('--output-file', '-o', help='Save total rewards to a file')
def run(env_name, input_code, avg, input_file, output_file):
    env = gym.make(env_name)
    observation_discretizer = bf.observation_discretizer(env.observation_space)
    action_sampler = bf.ActionSampler(env.action_space)

    if input_code:
        lines = [env_name]
    elif input_file:
        with open(input_file, 'r') as f:
            lines = list(f.readlines())
    else:
        raise ValueError("No program found. Specify it as a command line argument or input file")

    render = (avg == 1) and len(lines) < 5

    start_time = time.monotonic()
    if output_file:
        results = SortedDict()
    
    for line in lines:
        print(line)
        code = line.split(' ')[-1].strip()
        average_reward = 0
        error = False

        for idx in range(avg):
            program = bf.Program(code)
            executable = program.compile(observation_discretizer, action_sampler, cycle=True)
            rollout = executable.attend_gym(env, render = render)
            if executable.result in (bf.Result.SYNTAX_ERROR, bf.Result.STEP_LIMIT):
                error = True
                break
            average_reward += rollout.total_reward
            if render:
                print(f'Observations {rollout.states}')
                print(f'Actions {rollout.actions}')
                print(f'Total reward {rollout.total_reward}')

        if not error:
            average_reward /= avg
            print(f'Average reward {average_reward}')
            if output_file:
                results[average_reward] = line
        else:
            print('Error')

    print(f'{time.monotonic() - start_time} seconds elapsed')

    if output_file:
        with open(output_file, 'w') as f:
            f.writelines([f'{average_reward} {line}' for average_reward, line in results.items()])
                

if __name__ == '__main__':
    run()