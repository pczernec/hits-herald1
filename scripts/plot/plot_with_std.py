import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plot.load_log_data import load_log_data
from scripts.plot.utils import smooth_quantity
from scripts.config import ROOT_DIR



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training curves and store them in subdirectory 'plots' of experiment directory."
    )
    parser.add_argument("--env", nargs="+", type=str, help="Env name.")
    parser.add_argument("--methods", type=str, nargs="*", help="Method names.")
    parser.add_argument(
        "--experiment_dirs", type=str, nargs="*", help="Names of directories with logs."
    )
    parser.add_argument(
        "--plot_dir", type=str, help="Path to directory where plots will be created."
    )
    parser.add_argument(
        "--clip_steps", type=str, nargs='*', help="Clip number of steps to given value. Format: algo:steps", default=[]
    )
    parser.add_argument(
        "--smoothing_scale",
        type=float,
        help="Scale over which curves will be smoothed (in steps).",
        default=10000,
    )
    parser.add_argument(
        "--map_algos", type=str, nargs="*", default=[], help="map algos names"
    )
    parser.add_argument('--log_dir', type=str, help="Path to logs directory", default=os.path.join(ROOT_DIR, "data"))
    args = parser.parse_args()

    method_names = args.methods
    experiment_dirs_list = args.experiment_dirs
    plot_dir = os.path.join(ROOT_DIR, args.plot_dir)
    logs_dir = args.log_dir

    clip_steps = {env: int(steps) for env, steps in map(lambda x: x.split(':'), args.clip_steps)}
    algos_maps = {algo: mapto for algo, mapto in map(lambda x: x.split(':'), args.map_algos)}

    for env_name in args.env:

        logs_paths = {
            method_name: os.path.join(logs_dir, env_name, method_name, "log")
            for method_name in method_names
            if os.path.exists(os.path.join(logs_dir, env_name, method_name, "log"))
        }

        print(args)

        logs_dirs_dict = {
            method_name: [
                os.path.join(log_path, elem)
                for elem in (experiment_dirs_list if experiment_dirs_list else os.listdir(log_path))
                if os.path.exists(os.path.join(log_path, elem))
            ]
            for method_name, log_path in logs_paths.items()
        }

        fig = plt.figure(figsize=(5.5, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("steps")
        ax.set_ylabel("success rate")
        mode = "test"

        for i, (method_name, logs_dirs) in enumerate(logs_dirs_dict.items()):
            if logs_dirs == []:
                continue

            data_list = []
            for log_dir in logs_dirs:
                data_list.append(load_log_data(log_dir))


            results = []
            for data in data_list:
                steps = np.array(data["session"][mode]["step"])
                success = np.array(data["session"][mode]["success"])

                if env_name in clip_steps:
                    success = success[steps <= clip_steps[env_name]]
                    steps = steps[steps <= clip_steps[env_name]]

                sm_x, sm_y = smooth_quantity(steps, success, 50000, 1000)
                results.append(np.expand_dims(np.array(sm_y), 1))

            pal = sns.color_palette("colorblind")
            pal.as_hex()
            flatui = pal.as_hex()

            results = pd.DataFrame(
                np.concatenate(results, axis=1),
                columns=[f"run_{i}" for i in range(len(results))],
            )

            std = results.std(axis=1)
            mean = results.mean(axis=1)
            results["std"] = std
            results["mean"] = mean

            ax.plot(sm_x, results["mean"], lw=2, label=algos_maps.get(method_name, method_name), color=flatui[i])
            ax.fill_between(
                sm_x,
                results["mean"] + results["std"],
                results["mean"] - results["std"],
                facecolor=flatui[i],
                alpha=0.5,
            )

        ax.set_title(env_name)
        ax.legend(loc="upper left")
        ax.grid()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, env_name))
        plt.show()
        plt.close()
