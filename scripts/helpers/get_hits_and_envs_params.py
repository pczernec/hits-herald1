import json
import os
import sys
import pandas as pd
from gym import envs
import gym
import hac_envs
import dyn_rl_benchmarks


envs_names = [env for env in  os.listdir(os.path.join("data")) if not env.startswith(".")]
config_files = [
    os.path.join("data", env, "hits_trained", "graph_params.json") for env in envs_names
]

params = {}
for config_file, env_name in zip(config_files, envs_names):
    params[env_name] = {}
    with open(config_file) as json_file:
        graph_params = json.load(json_file)
        params[env_name] = graph_params["level_params_list"][1][
            "subtask_spec_params"
        ]["max_n_actions"]

registered = list(envs.registry.all())
used_envs = [env for env in registered if any([env_name in env.__str__() for env_name in envs_names])]
envs_steps = {env:env.max_episode_steps for env in used_envs}

env_steps_final = {}
for env_name in envs_names:
    print(env_name)
    env = gym.make(f"{env_name}-v1")
    env_steps_final[env_name] = {}
    try:
        print(env.env.max_episode_length)
        env_steps_final[env_name] = env.env.max_episode_length
    except:
        print("Not found")

env_steps_final["Pendulum"] = {}
env_steps_final["Pendulum"] = 200

df = pd.DataFrame([env_steps_final, params]).T

df["max_action_length"] = df[0]/df[1]
df.to_dict()