import argparse
import functools
import os
import json
import wandb

from scripts.config import ROOT_DIR
from scripts.run.train import override_params, create_wandb_config, create_wandb_name
from scripts.run.core import load_params, render, get_env_and_graph


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description = "Render episodes using final policy of run read from provided directory.")
    parser.add_argument("dir", default = None, help = "Directory containing parameters and final policy of run.")
    parser.add_argument("run_id", default=None, help="Wandb run id")
    parser.add_argument("--run_params", default=[], nargs="+", type=str)
    parser.add_argument("--test", default = False, action = "store_true", help = "Whether to show testing or training episode.")
    parser.add_argument("--learn", default = False, action = "store_true", help = "Whether to learn.")
    parser.add_argument("--wb_logs", default = False, action = "store_true", help = "Whether to use wandb for logging")
    parser.add_argument("--last_policy", default = False, action = "store_true", help = "Whether to automatically use the policy that the run was finished with")
    parser.add_argument("--no_render", default = False, action = "store_true", help = "Whether to render episodes.")
    parser.add_argument("--render_frequency", default = 1, type = int, help = "Frequency of rendered training episodes.")
    parser.add_argument("--do_not_load_policy", default = False, action = "store_true", help = "Whether to learn.")
    parser.add_argument("--torch_num_threads", default = None, type = int, help = "Overwrites number of threads to use in pytorch.")
    parser.add_argument("--tensorboard_logdir", default = None, help = "Directory for tensorboard log. Is created if necessary.")

    args = parser.parse_args()
    os.environ["rendering"] = "1"

    # load parameters from json files
    run_params, graph_params, varied_hps = load_params(args.dir)
    run_params_override = functools.partial(override_params, params=args.run_params)
    
    run_params_override(run_params)
    
    if args.torch_num_threads is not None:
        run_params["torch_num_threads"] = args.torch_num_threads

    # create environment and graph
    env, graph = get_env_and_graph(run_params, graph_params)

    # wandb logging
    wandb_params_path = os.path.join(
       args.dir, "wandb_params.json"
    )

    if args.wb_logs:
        try:
            with open(wandb_params_path) as json_file:
                wandb_params = json.load(json_file)
        except:
           raise Exception(f"No wandb_params file in {os.path.dirname(wandb_params_path)}")

        config_to_log = create_wandb_config(
            args.dir,
            override_run_params=run_params_override,
        )

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
        run_wandb.log_code()
    else:
        run_wandb = None

    # render
    render(args, graph, env, run_params, args.run_id, wandb_run_id=run_wandb.id if run_wandb else None)
