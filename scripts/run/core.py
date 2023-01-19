import os
import json
import pprint
import argparse
from typing import Optional, Any, Dict

import gym
import wandb
from gym.wrappers import Monitor
from wandb.sdk.wandb_run import Run

import graph_rl

import dyn_rl_benchmarks
import hac_envs
import reacting_rl_envs

from scripts.config import ROOT_DIR
from scripts.run.graphs import create_graph
from scripts.run.models import get_mlp_models
from scripts.run.subtask_spec_factories.string_to_subtask_spec_class import (
    get_subtask_spec_factory_class,
)


def load_params(dir, verbose=False):
    """Load parameters for run from json files.

    Does not integrate parameters of individual levels stored
    in separate files."""

    with open(os.path.join(dir, "run_params.json")) as json_file:
        run_params = json.load(json_file)

    with open(os.path.join(dir, "graph_params.json")) as json_file:
        graph_params = json.load(json_file)

    with open(os.path.join(dir, "varied_hp.json")) as json_file:
        varied_hps = json.load(json_file)

    if verbose:
        pp = pprint.PrettyPrinter(indent=4)
        print("Varied hyperparameters: ")
        pp.pprint(varied_hps)
        print("Run parameters:")
        pp.pprint(run_params)
        print("Graph parameters:")
        pp.pprint(graph_params)

    return run_params, graph_params, varied_hps


def graph_params_sanity_checks(graph_params):
    """Function that checks whether the run configuration is ok"""
    if len(graph_params["level_params_list"])==2 and graph_params["algorithm"] == "HERALD":
        assert "discounting_hl_gamma" in graph_params["level_params_list"][-1]['subtask_spec_params'], \
            "discounting_hl_gamma not specified"
        assert graph_params["level_params_list"][-1]['subtask_spec_params']["discounting_hl_gamma"] ==\
               graph_params["level_params_list"][-1]["algo_kwargs"]["flat_algo_kwargs"]["gamma"], \
            "Not matching HL gamma and discounting_hl_gamma"
        assert graph_params["level_params_list"][-1]['subtask_spec_params']["weight_delta_t_ach_aux"]==0,\
            "Wrong weight_delta_t_ach_aux != 0"
        assert graph_params["level_params_list"][-1]['subtask_spec_params']["use_env_reward"] and\
               graph_params["level_params_list"][-1]['subtask_spec_params']["use_env_reward_discounted"], \
            "Herald not using the environment reward"


def get_env_and_graph(run_params, graph_params, wandb_params=None):
    """Get env from gym and construct Graph_RL graph."""

    graph_params_sanity_checks(graph_params)

    env_name = run_params["env"]
    env_kwargs = run_params.get("env_kwargs", {})
    if isinstance(env_kwargs, str):
        env_kwargs = json.loads(env_kwargs)
    env = gym.make(env_name, **env_kwargs)

    if wandb_params is not None:
        if wandb_params["monitor_gym"]:
            os.makedirs(os.path.join(ROOT_DIR, "artifacts", "videos"), exist_ok=True)
            env = Monitor(
                env,
                os.path.join(ROOT_DIR, "artifacts", "videos", wandb_params["run_id"]),
                video_callable=lambda x: x % 200 == 0,
                force=True,
            )

    # specifiy subtask specs
    subtask_spec_cl_name = graph_params["subtask_spec_factory"]
    subtask_spec_cl = get_subtask_spec_factory_class(subtask_spec_cl_name)
    subtask_specs = subtask_spec_cl.produce(env, graph_params)

    # get models (actors, critics)
    level_algo_kwargs_list = get_mlp_models(graph_params["level_params_list"])

    # create graph
    graph = create_graph(
        env, graph_params, run_params, subtask_specs, level_algo_kwargs_list
    )

    return env, graph


def run_session(
    dir,
    graph,
    env,
    run_params,
    total_step_init=0,
    callback=None,
    wandb_params: Optional[Dict[str, Any]] = None,
    run_wandb: Optional[Run] = None,
    graph_params: Optional[Dict[str, Any]] = None,
    early_stopping = None
):
    """Run session for run and save resulting policy."""

    sess = graph_rl.Session(graph, env)

    # directory for saving the model parameters
    save_directory = os.path.join(dir, "model")
    if wandb_params is not None:
        save_directory = os.path.join(
            save_directory, f"{wandb_params['run_id']}"
        )
    os.makedirs(save_directory, exist_ok=True)

    if "model_save_frequency" in run_params:
        frequ = run_params["model_save_frequency"]

        class Cb_after_train_episode:
            def __init__(self):
                self.last_model_save = 0

            # callback is executed after each training episode
            def __call__(self, graph, sess_info, ep_return, graph_done):
                if sess_info.total_step - self.last_model_save >= frequ:
                    self.last_model_save = sess_info.total_step
                    # save model params
                    steps_in_k = int(sess_info.total_step / 1000)
                    save_path = os.path.join(save_directory, f"params_{steps_in_k}k.pt")
                    graph.save_parameters(save_path)
                    if wandb_params is not None and run_wandb is not None:
                        if (
                            "model_logging" in wandb_params
                            and wandb_params["model_logging"]
                        ):
                            trained_model_artifact = wandb.Artifact(
                                f'{graph_params["algorithm"]}_{run_wandb.id}',
                                type="model",
                            )
                            trained_model_artifact.add_file(save_path)
                            run_wandb.log_artifact(trained_model_artifact)

                # external part of callback
                if callback is not None:
                    callback(graph, sess_info, ep_return, graph_done)

        cb_after_train_episode = Cb_after_train_episode()
    else:
        cb_after_train_episode = None

    tensorboard_log = (
        False if "tensorboard_log" not in run_params else run_params["tensorboard_log"]
    )
    
    sess_props = sess.run(
        n_steps=run_params["n_steps"],
        max_runtime=run_params["max_runtime"] * 60.0
        if "max_runtime" in run_params
        else None,
        learn=True,
        render=False,
        test=True,
        test_render=bool(wandb_params["monitor_gym"]),
        tensorboard_logdir=os.path.join(dir, "tensorboard", str(run_wandb.id))
        if tensorboard_log
        else None,
        run_name=None,
        test_frequency=run_params["test_frequency"],
        test_episodes=run_params["n_test_episodes"],
        csv_logdir=os.path.join(dir, "log", run_wandb.id),
        torch_num_threads=run_params.get("torch_num_threads", None),
        append_run_name_to_log_paths=False,
        cb_after_train_episode=cb_after_train_episode,
        total_step_init=total_step_init,
        append_to_logfiles=total_step_init > 0,
        success_reward=run_params.get("success_reward", None),
        early_stopping = early_stopping
    )

    # save model params
    save_path = os.path.join(save_directory, "params.pt")
    graph.save_parameters(save_path)

    return sess_props


def render(args, graph, env, run_params, run_id, wandb_run_id):
    """Render episodes, learning and logging optional."""

    # load model params
    assert run_id is not None, "You need to provide a run_id"
    path = os.path.join(
        args.dir,
        "model",
    )
    models_path = [os.path.join(path, elem) for elem in os.listdir(path) if str(run_id) in elem][0]
    available_models = [elem for elem in os.listdir(models_path)]

    if not args.last_policy:
        print(f"Available models:")
        for i, model_name in enumerate(available_models):
            print(f"\t{i}: {model_name}")
        model_id = input(f"Give model id:")
        print(f"Chosen model: {available_models[int(model_id)]}")
    else:
        try:
            model_id = [available_models.index(e) for e in available_models if e.endswith("params.pt")][0]
            print(f"Chosen model: {available_models[int(model_id)]}")
        except IndexError:
            print("This run wasn't finished, --last_policy is invalid")
            print("Terminating")
            quit()

    matching_model = available_models[int(model_id)]
    load_path = os.path.join(models_path, matching_model)
    graph.load_parameters(load_path)     

    # run session
    sess = graph_rl.Session(graph, env)
    sess.run(
        n_steps=run_params["n_steps"],
        learn=args.learn,
        render=not args.no_render,
        test=args.test,
        test_render=not args.no_render,
        render_frequency=args.render_frequency,
        test_render_frequency=args.render_frequency,
        run_name=None,
        test_frequency=1,
        test_episodes=1,
        torch_num_threads=run_params.get("torch_num_threads", None),
        tensorboard_logdir=os.path.join(args.dir, "tensorboard", str(wandb_run_id) if wandb_run_id else str(run_id))
            if "tensorboard_log" in run_params else None,
        append_run_name_to_log_paths=False,
        train=not args.test,
        success_reward=run_params.get("success_reward", None),
    )
