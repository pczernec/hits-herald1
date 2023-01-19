import os
import optuna
from optuna.storages import RetryFailedTrialCallback
import argparse
import subprocess
from datetime import datetime
from multiprocessing import Pool
from tensorflow.python.summary.summary_iterator import summary_iterator

class EarlyStopping():

    def __init__(self, metrics_info, n_checkpoints, n_steps):
        self.metrics_info = metrics_info
        self.reader = summary_iterator
        self.check_time = int(n_steps / n_checkpoints)

    @staticmethod
    def make_metrics_info(values):
        metrics_info = {}
        for i in range(int(len(values)/3)):
            metrics_info[values[3*i]] = {'thr': float(values[3*i + 1]), 'type': values[3*i + 2]}
        return metrics_info
    
    @staticmethod
    def make_n_steps(run_params):
        return int([e for e in run_params if 'n_steps' in e][0].split('=')[1])

    def is_check_time(self, step):
        return True if step % self.check_time == 0 and step != 0 else False

    def checkpoint(self, path):
        print('Reading from: ', path) # path
        success = []
        for metric, info in self.metrics_info.items():
            values = []
            try:
                for event in self.reader(path):
                    for value in event.summary.value:
                        if value.tag == metric:
                            values.append(value.simple_value)
            except:
                print('DataLossError ommitted')
            if len(values) == 0:
                print(f'Value for {metric} not logged yet.')
                success.append(True)
                continue
            if info['type'] == 'mean':
                mean = sum(values) / len(values)
                print(f'Mean value of {metric}: ', mean)
                print(f'Threshold for this mean: ', info['thr'])
                if mean > info['thr']:
                    print('Threshold exceeded, checkpoint succeded!')
                else:
                    print('Checkpoint failed, aborting...')
                success.append(True if mean > info['thr'] else False)
            elif info['type'] == 'any':
                over_thr = [v > info['thr'] for v in values]
                print(f'Checking metric {metric}')
                if any(over_thr):
                    print(f'Threshold for {metric} exceeded, checkpoint succeeded!')
                else:
                    print('Chekpoint failed, aborting...')
                success.append(True if any(over_thr) else False)
        return all(success)

    def __bool__(self):
        return True

def _get_results_path(inputs):
    seed, hparams, args = inputs[0], inputs[1], inputs[2]
    print('Hparams names: ', hparams.keys())
    hparams_args = ['--graph_params'] + [
        f'{name}={value}' if name != 'level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma' else f'{name}={get_gamma(value)}' for name, value in hparams.items()]
    if 'level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma' in hparams.keys():
        hparams_args += [
            f"level_params_list[1].subtask_spec_params.discounting_hl_gamma={get_gamma(hparams['level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma'])}"]
        print(f"Chosen value for x in gamma = 1 - 10^x formula  is: {hparams['level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma']}")
        print(f"Gamma is equal to: {get_gamma(hparams['level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma'])}")
    if args.early_stopping:
        hparams_args += ['--early_stopping'] + args.early_stopping
        hparams_args += ['--n_checkpoints'] + [str(args.n_checkpoints)]
    print('Suffix: ', hparams_args)
    script_cmd = [
        'python', '-m' ,'scripts.run.train',
        '--hpo_mode',
        '--algo', args.algo,
        '--env', args.env,
        '--run_params', f'seed={seed}', f'n_steps={args.n_steps}'] + hparams_args
    cmd = subprocess.run(script_cmd, capture_output = True, universal_newlines=True)
    path = cmd.stdout.split('\n')[-1]
    return path

def _get_result(path, args):
    values = []
    print('Reading tb logs from: ', path)
    files = os.listdir(os.path.dirname(path))
    path = [os.path.join(os.path.dirname(path), file_name) for file_name in files if "tfevents" in file_name.casefold()][0]
    for event in summary_iterator(path):
        for value in event.summary.value:
                if value.tag == args.objective_name:
                    values.append(value.simple_value)
    return sum(values) / len(values)

def objective(trial, hp_dict, args):
    seeds = list(range(1, 2 * args.n_seeds + 1, 2))
    hparams = {}
    for name, properties in hp_dict.items():
        if properties['type'] == 'float':
            hparams[name] = trial.suggest_float(
                                name,
                                properties['lower_bound'], 
                                properties['upper_bound'])
        elif properties['type'] == 'int':
            hparams[name] = trial.suggest_int(
                                name,
                                properties['lower_bound'], 
                                properties['upper_bound'])
        elif properties['type'] == 'categorical':
            hparams[name] = trial.suggest_categorical(
                                name, 
                                properties['categories'])

    inputs = [(seed, hparams, args) for seed in seeds]

    with Pool(args.n_processes) as p:
        results_paths = p.map(_get_results_path, inputs)
    
    result_per_seed = [_get_result(path, args) for path in results_paths]
    
    return sum(result_per_seed) / len(result_per_seed)

def get_gamma(x):
    return 1 - 10**x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Optimize hyperparameters of a chosen policy and environment'
    )
    parser.add_argument(
        "--algo", 
        required = True, 
        help = "Which algorithm to use.")
    parser.add_argument(
        "--env",
        required = True,
        help = "Which environment to run.")
    parser.add_argument(
        '--graph_params_to_optimize', 
        nargs = "+",
        required = True,
        type = str,
        help = 'Which hyperparameters to tune. Must be provided in the same format as for scripts.run.train, e.g. level_params_list[1].subtask_spec_params.reward_normalization_coefficient.')
    parser.add_argument(
        '--objective_name',
        required = True,
        type = str,
        help = 'Which value to use as the objective for the optimization algorithm. Must be provided in the same format as tensorboard logs, e.g. env/success/test.'
    )
    parser.add_argument(
        '--n_seeds', 
        type = int,
        required = True,
        help = 'Number of seeds to use in each iteration. The formula used for calculating the seed is 2*i + 1 for i in {0, ..., --n_seeds - 1}, e.g. for --n_seeds==3, seeds 1, 3, 5 will be used.')
    parser.add_argument(
        '--n_steps',
        required = True,
        type = int,
        help = 'Number of maximum timesteps for each environment. This is required to remember that the number of steps differs between different environments.')
    parser.add_argument(
        '--minimize', 
        action = 'store_true', 
        default = False, 
        help = 'Whether to minimize the objective (default is to maximize)')
    parser.add_argument(
        '--n_trials',
        type = int,
        help = 'Number of iterations for the optimization algorithm to run, i.e. one trial consists of running one hyperparameter configuration per n_seeds number of seeds.',
        required = True
    )
    parser.add_argument(
        '--n_processes',
        default = 2,
        type = int,
        help = 'Number of processes to run in parallel',
        required = True
    )
    parser.add_argument(
        '--early_stopping', 
        default = [], 
        nargs = '+', 
        type = str
    )
    parser.add_argument(
        '--n_checkpoints', 
        default = 4, 
        type = int
    )
    parser.add_argument(
        '--resume',
        default = False,
        action = 'store_true',
        help = "Whether to resume an existing study. If provided, the user will be prompted for the name of the study (by default in the form of 'optuna study <date>')."
    )
    parser.add_argument(
        '--enqueue',
        default = False,
        action = 'store_true',
        help = "Whether to enqueue manually selected configuration in the beginning of the optimization process. If provided, the user will be prompted for the values of chosen hyperparameters."
    )
    parser.add_argument(
        '--add_trial',
        default = False,
        action = 'store_true',
        help = "Add a manually created trial to optimization process history."
    )
    args = parser.parse_args()

    hp_dict = {}
    for param in args.graph_params_to_optimize:
        print(f'Provide type of {param} hyperparameter (float, int, categorical):')
        t = input('Type: ')
        if t in ['int', 'float']:
            if param == 'level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma':
                print(f'Provide lower and upper bound for x in the following formula for gamma:')
                print('gamma = 1 - 10^x')
                l_b = float(input('Lower bound: '))
                u_b = float(input('Upper bound: '))
                print(f"The interval for gamma given by these bounds for x is: [{get_gamma(u_b)}, {get_gamma(l_b)}]")
            else:
                print(f'Provide lower and upper bound for {param}')
                l_b = float(input('Lower bound: '))
                u_b = float(input('Upper bound: '))
            hp_dict[param] = {
                'lower_bound': l_b,
                'upper_bound': u_b, 
                'type': t}
        elif t == 'categorical':
            categories = []
            n_cats = input('Provide number of categories: ')
            for i in range(int(n_cats)):
                categories.append(
                    input(f'Provide category name: ')
                )
                print(f'{i - 1} left')
            hp_dict[param] = {
                'type': t, 
                'categories': categories}

    study_name = input('Provide filename of the study database to resume (e.g. optuna study 2022-12-27 22:31:42.105569.db): ') if args.resume else f'optuna study {datetime.now()}'
    storage_name = f'sqlite:///{study_name}'
    
    storage = optuna.storages.RDBStorage(
        url=storage_name, 
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3))

    study = optuna.create_study(
        direction = 'minimize' if args.minimize else 'maximize',
        study_name = study_name,
        storage = storage, #storage_name
        load_if_exists = True if args.resume else False)
    print(f"direction = {'minimize' if args.minimize else 'maximize'}")

    if args.add_trial:
        params = {}
        distributions = {}
        n_params = int(input('Provide number of parameters for the trial that you want to add: '))
        for i in range(n_params):
            name = input('Provide parameter name: ')
            type = input('Type: ')
            value = input('Provide value: ')
            l_b = float(input('Lower bound: '))
            u_b = float(input('Upper bound: '))
            if type == 'float':
                params[name] = float(value)
                distributions[name] = optuna.distributions.FloatDistribution(
                    float(l_b), 
                    float(u_b))
            elif type == 'int':
                params[name] = int(value)
                distributions[name] = optuna.distributions.IntDistribution(
                    int(l_b), 
                    int(u_b))
        obj_value = float(input('Objective value for this configuration: '))
        trial_to_add = optuna.create_trial(
            params = params,
            distributions = distributions,
            value = obj_value
        )
        study.add_trial(trial_to_add) 

    if args.enqueue:
        trial_dict = {}
        for name in args.graph_params_to_optimize:
            if name == 'level_params_list[1].algo_kwargs.flat_algo_kwargs.gamma':
                trial_dict[name] = input(f'Provide specific value for {name} in terms of x in the gamma = 1 - 10^x formula: ')
                trial_dict[name] = float(trial_dict[name])
            else:
                trial_dict[name] = input(f'Provide specific value for {name}: ')
                if hp_dict[name]['type'] == 'float':
                    trial_dict[name] = float(trial_dict[name])
                elif hp_dict[name]['type'] == 'int':
                    trial_dict[name] = int(trial_dict[name])
        study.enqueue_trial(trial_dict)

    study.optimize(
        lambda trial: objective(trial, hp_dict = hp_dict, args = args), 
        n_trials = args.n_trials, 
        timeout = None)
