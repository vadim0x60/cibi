import click
import inspect
import os
import logging
from cibi.defaults import default_config_with_updates
from cibi.utils import get_dir_out_of_the_way

def task_launcher(f):
    relevant_options = set(inspect.getargspec(f).args)

    @click.command()
    @click.option('--config', default='', type=str, help='Configuration.')
    @click.option('--logdir', default='log', type=str, help='Absolute path where to write results.')
    @click.option('--task-id', default=0, type=int, help='ID for this worker.')
    @click.option('--num-workers', default=1, type=int, help='How many workers there are.')
    @click.option('--num-repetitions', default=1, type=int, help='Number of times the same experiment will be run (globally across all workers). Each run is independent.')
    @click.option('--log-level', default='INFO', type=str, help='The threshold for what messages will be logged. One of DEBUG, INFO, WARN, ERROR, or FATAL.')
    @click.option('--master', default='', type=str, help='URL of the TensorFlow master to use.')
    @click.option('--ps-tasks', default=0, type=int, help='Number of parameter server tasks. Only set to 0 for single worker training.')
    @click.option('--summary-interval', default=10, type=int, help='How often to write summaries.')
    @click.option('--summary-tasks', default=16, type=int, help='If greater than 0 only tasks 0 through summary_tasks - 1 will write summaries. If 0, all tasks will write summaries')
    @click.option('--do-profiling', is_flag=True, help='If True, cProfile profiler will run and results will be written to logdir. WARNING: Results will not be written if the code crashes. Make sure it exists successfully.')
    @click.option('--model-v', default=0, type=int, help='Model verbosity level.')
    @click.option('--delayed-graph-cleanup', is_flag=True, help='If true, container for n-th run will not be reset until the (n+1)-th run is complete. This greatly reduces the chance that a worker is still using the n-th container when it is cleared.')
    def run(**kwargs):
        kwargs['config'] = default_config_with_updates(kwargs['config'])
        
        logdir = kwargs['logdir']
        os.makedirs(logdir, exist_ok = True)
        parent_logger = logging.getLogger('cibi')
        parent_logger.setLevel(kwargs['log_level'])
        parent_logger.addHandler(logging.FileHandler(f'{logdir}/log.log'))

        kwargs = {k:v for k,v in kwargs.items() if k in relevant_options}
        if kwargs['num_repetitions'] == 1:
            get_dir_out_of_the_way(logdir)
            os.makedirs(logdir)
            f(**kwargs)
        else:
            for experiment_idx in range(kwargs['num_repetitions']):
                experiment_dir = os.path.join(logdir, f'exp{experiment_idx}')
                try:
                    os.makedirs(experiment_dir)
                except FileExistsError:
                    get_dir_out_of_the_way(experiment_dir)
                    os.makedirs(experiment_dir)

                kwargs['logdir'] = experiment_dir
                f(**kwargs)

    return run