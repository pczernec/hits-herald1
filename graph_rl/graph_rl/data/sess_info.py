from dataclasses import dataclass
from typing import Optional


@dataclass
class SessInfo:
    ep_step: Optional[int]
    total_step: Optional[int]
    learn: Optional[bool]
    testing: Optional[bool]

    def __init__(self, ep_step=None, total_step=None, learn=True, testing=False, rendering=False):
        self.ep_step = ep_step
        self.total_step = total_step
        self.learn = learn
        self.testing = testing
        self.rendering = rendering
        
