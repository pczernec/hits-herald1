import os
from abc import abstractmethod

import wandb

from dyn_rl_benchmarks import DrawbridgeEnv


def should_log():
    should_log = True
    if os.environ.get("rendering") is not None:
        rendering = os.environ["rendering"]
        if rendering == "1":
            should_log = False
    return should_log


class InterruptionLogger:
    def __init__(self, graph, env):
        self.graph = graph
        self.env = env

    @abstractmethod
    def check_and_log_interruption(self, sess_info):
        pass


class DummyLogger(InterruptionLogger):
    def __init__(self, graph, env):
        super().__init__(graph, env)

    def check_and_log_interruption(self, sess_info):
        pass


class InterruptionLoggerDrawbridge(InterruptionLogger):
    def __init__(self, graph, env):
        super().__init__(graph, env)

    def check_and_log_interruption(self, sess_info):
        interruption_times = self.graph._nodes[1].episode_info.interruption_times
        if isinstance(self.env.env, DrawbridgeEnv):
            random_element_timestep = self.env.env._get_random_element_timestep()
        else:
            return None

        delta_t = None
        # TODO: we should also not take into the consideration interruptions
        #  that are further than the start of the new HL action after bridge opening
        for i, interruption_time in enumerate(interruption_times):
            if interruption_time < random_element_timestep:
                continue
            else:
                delta_t = interruption_time - random_element_timestep
                break
        if should_log():
            wandb.log(
                {
                    "interruptions/delta_t": delta_t if delta_t else float('nan'),
                    "global_step": sess_info.total_step
                }
            )


def get_interruption_logger(env, graph) -> InterruptionLogger:
    if isinstance(env.env, DrawbridgeEnv):
        return InterruptionLoggerDrawbridge(graph, env)
    else:
        return DummyLogger(graph, env)
