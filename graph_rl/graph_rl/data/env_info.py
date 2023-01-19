from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class EnvInfo:
    reward: Optional[float]
    new_obs: Optional[Dict[str, Any]]
    done: Optional[bool]
    info: Optional[Dict[str, Any]]

    # for HERALD
    time_step: Optional[int]


    def __init__(self, reward = None, new_obs = None, done = None, info = None, time_step=None):

        self.reward = reward
        self.new_obs = new_obs
        self.done = done
        self.info = info

        # for HERALD
        self.time_step = time_step