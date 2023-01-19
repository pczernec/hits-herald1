import argparse
import json
import random
import shutil
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch
import os

from wandb.sdk.wandb_run import Run

# from graph_rl.utils.set_seed import init_global_seed_generator
from scripts.run.core import load_params, run_session, get_env_and_graph

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(
    dir_path,
    wandb_params=None,
    run_wandb: Optional[Run] = None,
    save_state: bool = False,
    override_graph_params: Optional[Callable[[Mapping[str, Any]], None]] = None,
    override_run_params: Optional[Callable[[Mapping[str, Any]], None]] = None,
    early_stopping = None
):
    # load parameters from json files
    run_params, graph_params, varied_hps = load_params(dir_path)
    if override_graph_params:
        override_graph_params(graph_params)
    if override_run_params:
        override_run_params(run_params)
    if os.path.isdir(os.path.join(dir_path, "state")):
        with open(os.path.join(dir_path, "state", "step.json"), "r") as json_file:
            step = json.load(json_file)["step"]
    else:
        step = 0

    print("Current step: ", step)

    log_dir = os.path.join(dir_path, "log")

    # seed python, numpy and pytorch
    seed = run_params["seed"]
    seed_everything(seed)

    env, graph = get_env_and_graph(run_params, graph_params, wandb_params)

    # load graph state in case the run has been executed before
    state_dir = os.path.join(dir_path, "state")
    if os.path.isdir(state_dir):
        print("Loading state of graph.")
        graph.load_state(dir_path=state_dir)
        # also override logs with saved version in case the process was killed
        # before the state could be saved
        log_state_dir = os.path.join(dir_path, "state", "log")
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        shutil.copytree(log_state_dir, log_dir)

    # run session
    sess_props = run_session(
        dir_path,
        graph,
        env,
        run_params,
        step,
        wandb_params=wandb_params,
        run_wandb=run_wandb,
        graph_params=graph_params,
        early_stopping = early_stopping
    )

    # delete old log directory in state directory
    log_state_dir = os.path.join(dir_path, "state", "log")
    if os.path.isdir(log_state_dir):
        shutil.rmtree(log_state_dir)

    if sess_props["timed_out"]:
        # save the state of the graph (replay buffer, parameters...) in
        # order to be able to continue training
        print("Saving state of graph.")
        state_dir = os.path.join(dir_path, "state")
        os.makedirs(state_dir, exist_ok=True)
        graph.save_state(os.path.join(state_dir))
        # save a copy of the log directory in the state directory
        shutil.copytree(os.path.join(dir_path, "log"), log_state_dir)
    else:
        # delete state if present
        state_path = os.path.join(dir_path, "state")
        if os.path.isdir(state_path):
            shutil.rmtree(state_path)

    if save_state:
        os.makedirs(os.path.join(dir_path, "state"), exist_ok=True)
        with open(os.path.join(dir_path, "state", "step.json"), "w") as json_file:
            json.dump({"step": sess_props["total_step"]}, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform run based on parameters read from json files in "
        "provided directory."
    )
    parser.add_argument(
        "dir",
        default=None,
        help="Directory with json files containing parameters for run.",
    )
    parser.add_argument(
        "--torch_num_threads",
        default=None,
        type=int,
        help="Overwrites number of threads to use in pytorch.",
    )
    args = parser.parse_args()

    run(args.dir, args.torch_num_threads)
