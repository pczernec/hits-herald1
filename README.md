# Hierarchical reinforcement learning with Timed Subgoals (HiTS)

This repository contains code for reproducing experiments from [our paper "Hierarchical reinforcement learning with Timed Subgoals"](https://proceedings.neurips.cc/paper/2021/hash/b59c21a078fde074a6750e91ed19fb21-Abstract.html). The implementation of the *Hierarchical reinforcement learning with Timed Subgoals* (HiTS) algorithm can be found in the [Graph-RL repository](https://github.com/nicoguertler/graph_rl). A short [video presentation](https://youtu.be/JkPaI3uZU6c) summarizes the algorithm as well as our experiments.

HiTS enables sample-efficient learning in sparse-reward, long-horizong tasks. In particular, it extends subgoal-based hierarchical reinforcement learning to environments with dynamic elements which are, most of the time, beyond the control of the agent. Due to the use of timed subgoals and hindsight action relabeling the higher level sees transitions that are consistent with a stationary effective environment. As a result both levels in the hierarchy can learn concurrently and efficiently.

The three benchmark tasks in dynamic environments from the paper are contained in the [dynamic-rl-benchmarks repository](https://github.com/martius-lab/dynamic-rl-benchmarks). If you are interested in applying HiTS to a different task, then [this demo](https://github.com/nicoguertler/graph_rl/blob/master/demos/hits_drawbridge_env.py) in the Graph-RL repository is the best place to start. 

## Installation

We recommend using a [virtual environment](https://docs.python.org/3/tutorial/venv.html) with python3.7 or higher. Make sure pip is up to date. In the root directory of the repository execute:

```bash
pip install -r requirements.txt
```

## Usage

To render episodes with one of the pretrained policies execute in the root directory:

```bash
python -m scripts.run.render --algo hits --env Platforms

```
Available algorithms:
* hits
* hac
* sac

Available environments:
* AntFourRooms
* Drawbridge
* Pendulum
* Platforms
* Tennis2D
* UR5Reacher

A policy can be be trained from scratch by running:

```bash
python -m scripts.run.train --algo hits --env Platforms

```

Remember to disable wandb using `WANDB_MODE=disabled` when debugging

Hyperparameters and seeds can be found in the `graph_params.json` files in the `data` directory. The key `level_params_list` contains a list of the hyperparameters of all levels, starting with the lowest level.

It is possible to override parameters from `graph_params.json` or `run_params.json` by using flags `--graph_params param=value \[param=value\]...` or `--run_params param=value \[param=value\]...`, e.g.:

```bash
python -m scripts.run.train --algo hits --env Platforms --graph_params level_params_list[0].model_kwargs.learning_rate=0.001 level_params_list[1].subtask_spec_params.max_n_actions=20
```

To render the episodes run, for instance:
```bash
python -m scripts.run.render_from_files /Users/michalbortkiewicz/repos/phd/rl/HiTS-master/data/NoisyPlatforms/herald_trained/ wyp5edcu --test --run_params env_kwargs.use_freeze=[False,False]
```

## Hyperparameter optimization using Optuna

You can search for the best hyperparameter configuration for a given policy and environment in an automated way using Optuna. For example, to tune the hyperparameters of HiTS on the Drawbridge environment, simply run:

```bash
python -m scripts.run.hpo_optuna --algo hits --env Drawbridge
```

with additional required arguments describing

1. which value should be used as the objective function, e.g.: 

```bash
--objective_name env/success/test_proper_reward
```

2. which hyperparameters you want to tune, e.g.:

```bash
--graph_params_to_optimize level_params_list[0].algo_kwargs.flat_algo_kwargs.gamma level_params_list[1].algo_kwargs.flat_algo_kwargs.alpha
```

You can also specify the details of the optimization process, i.e. choose the number of seeds with which the algorithm will be run in parallel (`--n_seeds`), the number of timesteps for each run (`--n_steps`), the number of trials to evaluate the algorithm (`--n_trials`), whether the objective should be minimized (`--minimize`, default is to maximize) and provide a default configuration to start with (`--enqueue`).

You may also apply an early stopping strategy. To do that, use `--early_stopping` followed by arguments in the following convention:

```bash
--early_stopping <name of the metric to track> <threshold value> <strategy type>
```

You can specify more than one strategy by adding arguments in this order. Currently available strategy types are `mean` and `any`. By using `mean`, everytime a checkpoint is performed (by default every 1/4 of `--n_steps`, can be adjusted with `--n_checkpoints`) the mean value of the specified metric will be calculated and compared with the `<threshold value>` - if it's not exceeded, the run is stopped. Type `any` checks whether, up to the checkpoint time, there is at least one value of the specified metric that exceeds the threshold and, if not, stops the run.

Note that tensorboard logging is required to perform hyperparameter tuning since it allows for tracking the objective value.

After running the script above, a database with the optimization history is created in the root directory of this repository and updated with each trial. You can follow the optimization process in real time by running:

```bash
optuna-dashboard sqlite:///<name of the .db file created>
```

## Old config (from HiTS) probably not well supported

### Tensorboard

To enable tensoboard logs, add the key

```json
"tensorboard_log": true
```

to the `run_params.json` file in the subdirectory of `data` corresponding to the desired environment. The tensorboard logs include returns, success rates, Q-values, number of actions used on each level before reaching the goal etc.

### Plotting 

Some quantities like returns and success rates are also logged in CSV files stored in the `data/*/log` directories. To plot them to PDFs, run

```bash
python -m scripts.plot.plot_training data/Environment/algo_trained/
```

with the desired environment and algorithm. The output files are written to `data/Environment/algo_trained/plots`.

Note that the logged return corresponds to the cumulative reward up to the point when the episode ended, either because the goal was reached, the environment timed out or the action budget of the higher level was exhausted. If the action budget of the higher level is finite, the return is therefore not a good indicator for learning progress. If the reward is negative, then the return might be high at the beginning of training because the subgoal budget is exhausted quickly. 

However, HiTS can be run without a finite subgoal budget. To run HiTS with this setting on Platforms, execute

```bash
python -m scripts.run.train --algo hits_no_budget --env Platforms
```

The logged returns will then be easier to interpret.

## Results

How HiTS outperforms baselines in dynamic environments can be seen in [this video](https://youtu.be/JkPaI3uZU6c?t=287).

## How to cite

Please use the following BibTex entry.

```
@article{gurtler2021hierarchical,
  title={Hierarchical Reinforcement Learning with Timed Subgoals},
  author={G{\"u}rtler, Nico and B{\"u}chler, Dieter and Martius, Georg},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
