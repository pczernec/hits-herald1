from dataclasses import dataclass
from typing import Optional


@dataclass
class ParentInfo:
    action: Optional[dict] = None
    algo_info: Optional[dict] = None
    step: Optional[int] = None
    hl_action_init_obs: Optional[dict] = None
    hl_action_horizon: Optional[int] = None
