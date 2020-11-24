import click
from cibi import bf
from cibi import bf_io
from cibi.extensions import make_gym
from sortedcontainers import SortedDict
import time

class ExecutionError(Exception):
    def __init__(self, result):
        super().__init__()
        self.result = result

def print_list(lname, l):
    print(lname)
    for elem in l:
        print(elem)

def run_episode(env, code, observation_discretizer, action_sampler, render=False, debug=False):
    executable = bf.Executable(code, observation_discretizer, action_sampler, cycle=True, debug=debug)
        
    rollout = executable.attend_gym(env, render = render)

    try:
        if executable.result in (bf.Result.SYNTAX_ERROR, bf.Result.STEP_LIMIT):
            raise ExecutionError(executable.result)
    except AttributeError:
        pass

    if debug:
        print_list('Trace:', executable.program_trace)
        print_list('Observation trace:', observation_discretizer.trace)
        print_list('Action trace:', action_sampler.trace)
        print(f'Total reward {rollout.total_reward}')
        
    return rollout

def average(coll):
    return sum(coll) / len(coll)

@click.command()
@click.argument('env_name', required=True, type=str)
@click.argument('input_code', required=False, type=str)
@click.option('--avg', help='Average results over multiple runs', type=int, default=1)
@click.option('--best', help='Take the best result over multiple runs', type=int, default=1)
@click.option('--force-fluid-discretization', help='Use fluid discretization even if an observation has lower and upper bounds defined', is_flag=True)
@click.option('--fluid-discretization-history', help='Length of the observation history kept for fluid discretization', type=int, default=1024)
@click.option('--input-file', '-i', help='Load the programs from a specified file')
@click.option('--output-file', '-o', help='Save total rewards to a file')
@click.option('--debug', is_flag=True, help='Show a visual rendering of the env and log full execution traces')
def run(env_name, input_code, avg, best, input_file, output_file, 
        debug, force_fluid_discretization, fluid_discretization_history):
    env = make_gym(env_name)

    if input_code:
        lines = [input_code]
    elif input_file:
        with open(input_file, 'r') as f:
            lines = list(f.readlines())
    else:
        raise ValueError("No program found. Specify it as a command line argument or input file")

    render = debug or ((avg == 1) and (best == 1) and len(lines) < 5)

    start_time = time.monotonic()
    if output_file:
        results = SortedDict()
    
    for line in lines:
        print(line)
        code = line.split(' ')[-1].strip()

        observation_discretizer = bf_io.ObservationDiscretizer(env.observation_space, debug=debug, 
                                                               force_fluid=force_fluid_discretization,
                                                               history_length=fluid_discretization_history)
        action_sampler = bf_io.ActionSampler(env.action_space, debug=debug)

        random_agent = bf.Executable('@!', observation_discretizer, action_sampler, cycle=True, debug=False)
        episode_count =bf_io.burn_in(env, random_agent, observation_discretizer, action_sampler)
        print(f'{episode_count} episodes of burn in done')

        if debug:
            print(f'Discretization thresholds: {observation_discretizer.get_thresholds()}')
        
        try:
            average_best_reward = average([max(run_episode(env, code, observation_discretizer, action_sampler, render, debug).total_reward 
                                               for best_idx in range(best)) for avg_idx in range(avg)])
            print(f'Average best reward {average_best_reward}')
            if output_file:
                results[average_best_reward] = line
        except ExecutionError:
            print('Error')
            
    print(f'{time.monotonic() - start_time} seconds elapsed')

    if output_file:
        with open(output_file, 'w') as f:
            f.writelines([f'{average_reward} {line}' for average_reward, line in results.items()])
                

if __name__ == '__main__':
    run()