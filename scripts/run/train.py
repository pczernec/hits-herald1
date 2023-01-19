import argparse
import ast
import functools
import json
import os
import sys
from typing import Any, Iterable, Mapping, Optional, Callable

import wandb


from scripts.config import ROOT_DIR
from scripts.run.run_from_files import run
from scripts.run.hpo_optuna import EarlyStopping


def create_wandb_config(
    configs_dir_path: str,
    override_graph_params: Optional[Callable[[Mapping[str, Any]], None]] = None,
    override_run_params: Optional[Callable[[Mapping[str, Any]], None]] = None,
):
    configs_paths = [
        os.path.join(configs_dir_path, elem)
        for elem in os.listdir(configs_dir_path)
        if elem.endswith(".json") and "varied_hp" not in elem
    ]
    configs = []
    for config_path in configs_paths:
        with open(config_path) as json_file:
            config = json.load(json_file)
            if "graph" in config_path:
                if override_graph_params:
                    override_graph_params(config)
            if "run" in config_path:
                if override_run_params:
                    override_run_params(config)
            configs.append(config)
    super_dict = {}
    for d in configs:
        for k, v in d.items():
            super_dict[k] = v
    return super_dict


def override_params(config: Mapping[str, Any], params: Iterable[str]):
    for param in params:
        whole_key, str_value = param.split("=", maxsplit=1)
        key = whole_key
        try:
            value = ast.literal_eval(str_value)
        except:
            value = str_value
        conf = config
        while key:
            if key.startswith("["):
                assert "]" in key
                ind_str, key = key.split("]", 2)
                key = key.lstrip(".")
                ind = int(ind_str[1:])

                assert ind < len(conf), f"{subkey} not found in ({whole_key})"

                if key:
                    conf = conf[ind]
                else:
                    conf[ind] = value
            elif "[" in key and ("." not in key or key.index("[") < key.index(".")):
                subkey, key = key.split("[", 2)
                key = "[" + key
                assert subkey in conf, f"{subkey} not found in ({whole_key})"
                conf = conf[subkey]
            elif "." in key:
                keys = key.split(".", maxsplit=1)
                subkey, key = keys
                assert subkey in conf, f"{subkey} not found in ({whole_key})"
                conf = conf[subkey]
            else:
                assert key in conf, f"{key} not found in ({whole_key})"
                conf[key] = value
                key = None


def set_name_params(name, current_key, value):
    if isinstance(value, Mapping):
        for key, v in value.items():
            name = set_name_params(
                name, current_key + "." + key if current_key else key, v
            )
    elif isinstance(value, Iterable) and not isinstance(value, str):
        for i, v in enumerate(value):
            name = set_name_params(name, f"{current_key}[{i}]", v)
    name = name.replace(f"{{{current_key}}}", str(value))
    return name


def create_wandb_name(name, config):
    return set_name_params(name, None, config)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train policy on given environment using specified algorithm."
    )
    parser.add_argument(
        "--algo", default="hits", help="Which algorithm to use (hac or hits)."
    )
    parser.add_argument(
        "--env",
        default="Platforms",
        help="Which environment to run (Platforms, Drawbridge or Tennis2D).",
    )
    parser.add_argument(
        "--save-state",
        default=False,
        type=bool,
        help="Whether to save state after full training.",
    )
    parser.add_argument("--graph_params", default=[], nargs="+", type=str)
    parser.add_argument("--run_params", default=[], nargs="+", type=str)
    parser.add_argument("--hpo_mode", default = False, action = "store_true")
    parser.add_argument("--early_stopping", default = [], nargs = '+', type = str)
    parser.add_argument("--n_checkpoints", default = 4, type = int)

    args = parser.parse_args()
    assert args.algo in {"hits", "hits_no_budget", "hac", "sac", "herald"}
    assert args.env in {
        "AntFourRooms",
        "Drawbridge",
        "NoisyDrawbridge",
        "Pendulum",
        "Platforms",
        "NoisyPlatforms",
        "EasyPlatforms",
        "TwoNoisyPlatforms",
        "Tennis2D",
        "UR5Reacher",
        "HitTarget",
        "BooleanDrawbridge",
        "CrashDrawbridge",
        "ThreePlatforms"
    }
    print(f"Training with {args.algo} on {args.env} environment.")

    wandb_params_path = os.path.join(
        f"{ROOT_DIR}/data", args.env, args.algo + "_trained", "wandb_params.json"
    )

    try:
        with open(wandb_params_path) as json_file:
            wandb_params = json.load(json_file)
    except:
        raise Exception(f"No wandb_params file in {os.path.dirname(wandb_params_path)}")

    path = os.path.join(f"{ROOT_DIR}/data", args.env, args.algo + "_trained")

    graph_params_override = functools.partial(override_params, params=args.graph_params)
    run_params_override = functools.partial(override_params, params=args.run_params)

    config_to_log = create_wandb_config(
        path,
        override_graph_params=graph_params_override,
        override_run_params=run_params_override,
    )

    if wandb_params is not None:
        run_wandb = wandb.init(
            dir=os.path.join(ROOT_DIR),
            project=wandb_params["project"],
            entity=wandb_params["entity"],
            name=create_wandb_name(wandb_params["name"], config_to_log),
            group=wandb_params["group"],
            tags=wandb_params["tags"] if "tags" in wandb_params else [],
            sync_tensorboard=wandb_params["sync_tensorboard"],
            monitor_gym=wandb_params["monitor_gym"],
            config=config_to_log,
            save_code=wandb_params["save_code"],
        )
        wandb_params["run_id"] = run_wandb.id
    else:
        run_wandb = None

    if args.early_stopping:
        metrics_info = EarlyStopping.make_metrics_info(args.early_stopping)
        n_steps = EarlyStopping.make_n_steps(args.run_params)
        early_stopping = EarlyStopping(
            metrics_info = metrics_info,
            n_checkpoints = args.n_checkpoints,
            n_steps = n_steps
        )
    else:
        early_stopping = None

    run_wandb.log_code()
    run(
        path,
        wandb_params=wandb_params,
        run_wandb=run_wandb,
        save_state=args.save_state,
        override_graph_params=graph_params_override,
        override_run_params=run_params_override,
        early_stopping = early_stopping
    )

    if args.hpo_mode:
        path_files = run_wandb.dir
        name_tf_events = [e for e in os.listdir(path_files) if e.startswith('events')][0]
        path_tf_events = os.path.join(path_files, name_tf_events)
        print('Tb logs at: ', path_tf_events)
        sys.stdout.write(path_tf_events)
