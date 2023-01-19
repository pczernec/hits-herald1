from typing import Callable
import tianshou as ts
import numpy as np
import torch


class VariableDiscountSACPolicy(ts.policy.SACPolicy):
    @staticmethod
    def compute_nstep_return(
        batch: ts.data.Batch,
        buffer: ts.data.ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ts.data.ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> ts.data.Batch:
        assert n_step == 1, "VariableDiscountSACPolicy requires n_step to be 1"
        gamma = (gamma ** batch.info['act_length']).reshape(-1, 1)
        return ts.policy.BasePolicy.compute_nstep_return(batch, buffer, indice, target_q_fn, gamma, n_step, rew_norm)
