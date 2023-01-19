import os
from typing import Optional, Dict, Any
import inspect

import numpy as np
import wandb

from scripts.run.loggers import should_log
from . import Graph
from ..nodes import HACNode
from ..nodes.herald_node import HeraldNode
from ..utils import listify
from ..aux_rewards import EnvReward


class HeraldGraph(Graph):

    def __init__(
        self,
        name,
        n_layers,
        env,
        subtask_specs,
        HAC_kwargs,
        HiTS_kwargs,
        HERALD_kwargs,
        update_sgs_rendering=None,
        update_tsgs_rendering=None,
        env_reward_weight=None,
        node_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
        n_layers (int): Number of layers in hierarchy.
        """

        self.hl_actions_without_interruptions_prev = 0
        self.n_layers = n_layers
        self._update_sgs_rendering = update_sgs_rendering
        self._update_tsgs_rendering = update_tsgs_rendering

        # convert parameters to list with one entry per layer if not already a list
        HiTS_kwargs_list = listify(HiTS_kwargs, n_layers - 1)
        subtask_specs = listify(subtask_specs, n_layers - 1)

        # by default penalty for choosing a goal which was not reached by the child is minus
        # maximum number of actions in episode on this layer
        if "child_failure_penalty" not in HAC_kwargs:
            HAC_kwargs["child_failure_penalty"] = -subtask_specs[-1].max_n_actions
        for kwargs, subtask_spec in zip(HiTS_kwargs_list, subtask_specs):
            if "child_failure_penalty" not in kwargs:
                kwargs["child_failure_penalty"] = -1.0

        # for emitting timed subgoals if desired
        if env_reward_weight is not None:
            subtask_specs[-1].add_aux_reward(EnvReward(env_reward_weight))

        # highest node is HAC node because most environments do not provide a
        # timed subgoal
        highest_node = HACNode(
            name="hac_node_layer_{}".format(n_layers - 1),
            parents=[],
            subtask_spec=subtask_specs[-1],
            HAC_kwargs=HAC_kwargs,
            delta_t_max=subtask_specs[0]._delta_t_max
        )

        # construct graph
        parents = [highest_node]
        self._nodes = [highest_node]
        for l in range(n_layers - 2, -1, -1):
            node_name = "herald_node_layer_{}".format(l)
            node = HeraldNode(
                name=node_name,
                parents=parents,
                subtask_spec=subtask_specs[l],
                HiTS_kwargs=HiTS_kwargs_list[l],
                HERALD_kwargs=HERALD_kwargs,
                env=env,
                node_kwargs=node_kwargs,
            )
            if l < n_layers - 1:  # TODO: tautology
                self._nodes[0].add_child(node)
            self._nodes.insert(0, node)
            parents = [node]

        entry_node = self._nodes[-1]  # the last one is root

        super(HeraldGraph, self).__init__(name, entry_node, env.action_space)

    def get_atomic_action(self, env_obs, sess_info, start_node=None, testing=False):
        action = super().get_atomic_action(env_obs, sess_info, start_node, testing)

        # update timed subgoals via callback for rendering them in environment
        if self._update_tsgs_rendering is not None:
            subgoals = [nd.current_timed_goal for nd in self._nodes[:-1]]
            tolerances = [nd.current_goal_tol for nd in self._nodes[:-1]]
            if "interrupted" in inspect.getfullargspec(self._update_tsgs_rendering)[0] and self._nodes[0]._interrupt:
                self._update_tsgs_rendering(subgoals, tolerances, interrupted=True)
            else:
                self._update_tsgs_rendering(subgoals, tolerances)
        return action

    def log_additional_info_wandb(self, sess_info=None):
        """Log additional info about interruptions"""

        n_hl_actions_taken = self._nodes[1].algorithm._actions_taken - self.hl_actions_without_interruptions_prev
        self.hl_actions_without_interruptions_prev = self._nodes[1].algorithm._actions_taken

        avg_interrupted_actions_predicted_len = np.mean(
            self._nodes[1].episode_info.actions_interrupted_len_predicted) \
            if len(self._nodes[1].episode_info.actions_interrupted_len_predicted) > 0 else float('nan')
        avg_interrupted_actions_actual_len = np.mean(
            self._nodes[1].episode_info.actions_interrupted_len_actual) \
            if len(self._nodes[1].episode_info.actions_interrupted_len_actual) > 0 else float('nan')

        if should_log():
            wandb.log(
                {"interruptions/n_interruptions": self._nodes[1].episode_info.interruptions
                if sess_info.total_step > self._nodes[0].interruption_helper.interruption_start else 0
                    , "global_step": sess_info.total_step}
            )
            wandb.log(
                {"interruptions/n_actions_sum": self._nodes[1].episode_info.interruptions + n_hl_actions_taken
                if sess_info.total_step > self._nodes[0].interruption_helper.interruption_start else 0
                    , "global_step": sess_info.total_step}
            )
            wandb.log(
                {"interruptions/interrupted_actions_len_pred": avg_interrupted_actions_predicted_len
                if sess_info.total_step > self._nodes[0].interruption_helper.interruption_start else float('nan')
                    , "global_step": sess_info.total_step}
            )
            wandb.log(
                {"interruptions/interrupted_actions_len_actual": avg_interrupted_actions_actual_len
                if sess_info.total_step > self._nodes[0].interruption_helper.interruption_start else float('nan')
                    , "global_step": sess_info.total_step}
            )
