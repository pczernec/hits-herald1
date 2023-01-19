# import os
# import random
# from typing import Any, Callable, Generator, Optional, TypeVar, cast
#
# import numpy as np
# import torch
# from gym.utils.seeding import create_seed
# import csv
#
# from scripts.config import ROOT_DIR
#
# F = TypeVar('F', bound=Callable[..., Any])
#
# GRAPH_RL_SEED: Optional[int] = None
#
#
# def seed_generator() -> Generator[int, int, None]:
#     global GRAPH_RL_SEED
#     if GRAPH_RL_SEED is not None:
#         seed = GRAPH_RL_SEED
#     else:
#         raise ValueError('Global variable `GRAPH_RL_SEED` is not set!')
#     while True:
#         total_step = yield create_seed(seed)
#         seed = GRAPH_RL_SEED + total_step
#         seed = int(seed) % (2**32-1)
#
#
# # Global variable
# global_seed_generator = seed_generator()
#
#
# def set_global_seed(total_step: int) -> None:
#     seed = global_seed_generator.send(total_step)
#
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#
# def reset_seed(func: F) -> F:
#     def new_func(*args, **kwargs):
#         # If total_step is 0 then pass None for the generator initialization
#         if 'sess_info' in kwargs:
#             total_step = kwargs['sess_info'].total_step + 1  # +1 to not get same seed after init
#         else:
#             total_step = args[4].total_step + 1
#         set_global_seed(total_step)
#
#         # NOTE: Use when need to log the random function etc
#         # try:
#         #     delta_t = args[2].action['delta_t_ach'][0]
#         # except Exception:
#         #     delta_t=""
#         # with open(os.path.join(ROOT_DIR, "artifacts", "function_call_logs", "herald3.csv"), 'a') as f:
#         #     writer = csv.writer(f)
#         #     writer.writerow([str(total_step), type(args[0]).__name__, np.random.get_state()[1][0], delta_t])
#
#         return func(*args, **kwargs)
#
#     return cast(F, new_func)
#
#
# def init_global_seed_generator(seed: int) -> int:
#     global GRAPH_RL_SEED
#     GRAPH_RL_SEED = seed
#     return next(global_seed_generator)
