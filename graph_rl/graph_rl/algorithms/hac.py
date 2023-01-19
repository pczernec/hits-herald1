import csv
import os
from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from ..data import FlatTransition
from . import OffPolicyAlgorithm
from .goal_sampling_strategy import GoalSamplingStrategy
from ..nodes import Node
from ..subtasks.shortest_path_subtask import ChildInfo


class HAC(OffPolicyAlgorithm):
    """Hierarchical Actor-Critic."""

    supported_goal_sampling_strategies = {"future", "episode", "final", "future_and_now"}

    def __init__(self, name, model, child_failure_penalty, check_achievement,
                 flat_algo_kwargs=None, flat_algo_name="SAC", goal_sampling_strategy="future",
                 buffer_size=100000, batch_size=128, n_hindsight_goals=4,
                 testing_fraction=0.3, fully_random_fraction=0.1, bootstrap_testing_transitions=True,
                 use_normal_trans_for_testing=False, use_testing_transitions=True,
                 learn_from_deterministic_episodes=True, log_q_values=True, learning_starts=0,
                 bootstrap_end_of_episode=True, grad_steps_per_env_step=1, include_interrupted_actions_to_budget=False,
                 delta_t_max=None):
            """
            Args:
                check_achievement (callable): Maps achieved_goal,
                    desired_goal and parent_info to boolean achieved and float
                    reward indicating whether the achieved_goal satisfies the
                    desired goal and what reward this implies.
                """

            super(HAC, self).__init__(name, flat_algo_name, model, fully_random_fraction,
                    flat_algo_kwargs, buffer_size, batch_size, learning_starts,
                    grad_steps_per_env_step)

            self._delta_t_max = delta_t_max
            assert goal_sampling_strategy in self.supported_goal_sampling_strategies\
                    or isinstance(goal_sampling_strategy, GoalSamplingStrategy), \
                    "Goal sampling strategy {} not supported.".format(goal_sampling_strategy)

            self._child_failure_penalty = child_failure_penalty
            self._goal_sampling_strategy = goal_sampling_strategy
            self._check_achievement = check_achievement
            self._n_hindsight_goals = n_hindsight_goals
            self._testing_fraction = testing_fraction
            self._bootstrap_testing_transitions = bootstrap_testing_transitions

            self._use_normal_trans_for_testing = use_normal_trans_for_testing
            self._use_testing_transitions = use_testing_transitions
            self._learn_from_deterministic_episodes = learn_from_deterministic_episodes
            self._log_q_values = log_q_values
            self._bootstrap_end_of_episode = bootstrap_end_of_episode

            self._verbose = False
            self._debug = False
            self.include_interrupted_actions_to_budget = include_interrupted_actions_to_budget

    def _sample_achieved_goals(self, current_index):

        goals = []
        if self._n_hindsight_goals > 0:
            n_transitions = len(self._episode_transitions)
            if self._goal_sampling_strategy == "future_and_now":
                n_goals = min(self._n_hindsight_goals,
                        n_transitions - current_index)
                indices = np.random.randint(low = current_index, high = n_transitions,
                        size = n_goals)
            if self._goal_sampling_strategy == "future":
                n_goals = min(self._n_hindsight_goals,
                        n_transitions - current_index - 1)
                indices = np.random.randint(low = current_index + 1, high = n_transitions,
                        size = n_goals)
            elif self._goal_sampling_strategy == "episode":
                n_goals = min(self._n_hindsight_goals, n_transitions)
                indices = np.random.randint(low = 0, high = n_transitions, size = n_goals)
            elif self._goal_sampling_strategy == "final":
                # only generate hindsight goal from final state if environment is done
                if self._episode_transitions[-1].env_done:
                    indices = [-1]
                else:
                    indices = []
            elif isinstance(self._goal_sampling_strategy, GoalSamplingStrategy):
                indices = self._goal_sampling_strategy(self._episode_transitions,
                        self._n_hindsight_goals)

            for i in indices:
                goals.append(self._episode_transitions[i].subtask_tr.info["achieved_generalized_goal"])

        return goals

    def _add_experience_to_flat_algo(self, parent_info, deterministic_episode, node_is_sink, sess_info, **kwargs):
        if not sess_info.testing:
            if self._learn_from_deterministic_episodes or not deterministic_episode:
                self._ep_return = 0
                # add transitions of this episode to replay buffer of flat RL algorithm
                # (by applying hindsight goal and action manipulations)
                for trans_index, tr in enumerate(self._episode_transitions):
                    # TODO: refactor - because it it probably misleading mechanism
                    has_achieved_now = tr.subtask_tr.info.get("has_achieved", False)

                    if self._bootstrap_end_of_episode:
                        done = has_achieved_now
                    else:
                        done = has_achieved_now or tr.env_info.done

                    # unaltered flat transition
                    f_trans_0 = FlatTransition(
                        obs = tr.subtask_tr.obs,
                        action = tr.subtask_tr.action,
                        reward = tr.subtask_tr.reward * tr.child_feedback.get('reward_part', 1) if tr.child_feedback else tr.subtask_tr.reward,
                        new_obs = tr.subtask_tr.new_obs,
                        done = done,
                        info = {'act_length': tr.subtask_tr.info['act_length']})
                    f_trans_base = f_trans_0

                    # if the node is a sink, do not attempt to manipulate action
                    # in hindsight and add original transition to replay buffer
                    if node_is_sink:
                        self._add_to_flat_replay_buffer(f_trans_0)
                        self._ep_return += f_trans_0.reward

                    # if the node is not a sink consider testing transitions
                    # and hindsight action transitions
                    else:
                        # boolean indicating whether the child node used a deterministic version of its policy
                        is_testing_transition = tr.algo_info["child_be_deterministic"]

                        # boolean indicating whether child achieved subgoal
                        did_child_achieve_subgoal = tr.child_feedback["has_achieved"]

                        if not did_child_achieve_subgoal:
                            if is_testing_transition or self._use_normal_trans_for_testing:
                                # Penalty for HL policy if not realizable sub-goals for LL
                                f_trans_testing = self._hl_action_testing_transition(
                                    f_trans_0, tr
                                )
                                if f_trans_testing:  # it could be FlatTransition or None if particular flag is on
                                    self._add_to_flat_replay_buffer(f_trans_testing)
                                    # Note: should we add this to ep_return?
                                    self._ep_return += f_trans_testing.reward

                            # We relabel action so that we act as LL policy is optimal
                            f_trans_base = self._hl_action_hindsight(
                                 f_trans_0, tr
                            )
                        self._add_to_flat_replay_buffer(f_trans_base)
                        self._ep_return += f_trans_0.reward

                    # HER - hindsight goal transitions (based on hindsight action
                    # transition or original transition in case of a sink node)
                    achieved_goals = self._sample_achieved_goals(trans_index)
                    for hindsight_goal in achieved_goals:
                        f_trans_hindsight_goal = deepcopy(f_trans_base)
                        f_trans_hindsight_goal.obs = {
                            "partial_observation": tr.subtask_tr.obs["partial_observation"],
                            "desired_goal": hindsight_goal
                        }
                        f_trans_hindsight_goal.new_obs = {
                            "partial_observation": tr.subtask_tr.new_obs["partial_observation"],
                            "desired_goal": hindsight_goal
                        }
                        achieved, f_reward = self._check_achievement(
                            achieved_goal = tr.subtask_tr.info["achieved_generalized_goal"],
                            desired_goal = hindsight_goal,
                            obs = f_trans_hindsight_goal.obs,
                            action = f_trans_hindsight_goal.action,
                            parent_info = parent_info,
                            env_info = tr.env_info,
                            child_info=tr.subtask_tr.info.get('child_info', None)
                        )
                        f_trans_hindsight_goal.done = achieved and tr.child_feedback.get("final_step", True) if tr.child_feedback else achieved
                        f_trans_hindsight_goal.reward = f_reward * tr.child_feedback.get('reward_part', 1) if tr.child_feedback else f_reward
                        self._add_to_flat_replay_buffer(f_trans_hindsight_goal)

                # log undiscounted ep return to tensorboard (includes contribution
                # from testing transitions but not from hindsight transitions!)
                # tensorboard logging
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(f"{self.name}/ep_return", self._ep_return, sess_info.total_step)

        self._last_rewards = [elem.env_info.reward for elem in self._episode_transitions]
        self._episode_transitions.clear()

    def _hl_action_hindsight(self, f_trans_0, tr):
        """HL action hindsight relabels the action so that it matches the achieved goal."""

        # hindsight action transition
        # If child node achieved desired goal use original action.
        # If not, use subgoal the child achieved as action.
        f_trans_hindsight_action = deepcopy(f_trans_0)
        f_trans_hindsight_action.action = tr.child_feedback["achieved_generalized_goal"]
        # TODO: In principle, reward could depend on action, so have to recompute the reward here.
        # TODO: take care of it!!!
        if self._verbose:
            print(
                "hindsight action transition in {}\n".format(self.name)
                + str(f_trans_hindsight_action)
            )
        return f_trans_hindsight_action

    def _hl_action_testing_transition(self, f_trans_0, tr):
        """Testing transitions check whether sub-policy achieved desired goal and penalize failure."""

        # add testing transitions (if enabled)
        # Optionally, also transitions generated by stochastic
        # lower level are used as testing transitions.
        # Only add transition with penalty if child node failed
        # to achieve its subgoal, otherwise do not add any transition.

        f_trans_testing = None
        if self._use_testing_transitions:
            f_trans_testing = deepcopy(f_trans_0)
            if tr.algo_info["interrupted"]:
                if f_trans_0.done and self._delta_t_max:  # Don't give penalty when achieved env goal while using Herald
                    pass
                else:
                    penalty_scaling_factor = 1
                    if self._delta_t_max:
                        penalty_scaling_factor = self._delta_t_max
                    f_trans_testing.reward = self._child_failure_penalty * tr.child_feedback.get(
                        "reward_part", 1
                    ) * f_trans_0.info['act_length'] / penalty_scaling_factor

            else:
                f_trans_testing.reward = self._child_failure_penalty * tr.child_feedback.get(
                    "reward_part", 1
                )
            if not self._bootstrap_testing_transitions:
                # Have to set done to True in these transitions
                # (according to HAC paper and accompanying code).
                f_trans_testing.done = True
            if self._verbose:
                print("testing transition in {}\n".format(self.name) + str(f_trans_testing))
        return f_trans_testing

    def add_experience(
            self,
            env_obs,
            env_info,
            subtask_trans,
            parent_info,
            child_feedback,
            algo_info,
            sess_info,
            task_spec=None,
            propagate_child_feedback=False,
            child_info: Optional[ChildInfo] = None
    ):
        super().add_experience(
            env_obs,
            env_info,
            subtask_trans,
            parent_info,
            child_feedback,
            algo_info,
            sess_info,
            child_info,
        )

        # So that we do not count this interrupted action as budget use
        if not self.include_interrupted_actions_to_budget:
            if algo_info['interrupted']:
                self._actions_taken -= 1

        if algo_info['interrupted'] and not (subtask_trans.ended or env_info.done):
            if parent_info:
                deterministic_episode = algo_info["is_deterministic"]
            else:
                deterministic_episode = False

            node_is_sink = child_feedback is Node.sink_identification_feedback
            self._add_experience_to_flat_algo(
                parent_info, deterministic_episode, node_is_sink, sess_info
            )

    def get_algo_info(self, env_obs, parent_info, force_deterministic=False):
        # child_be_deterministic instructs the children to be deterministic whereas
        # is_deterministic implies that this node is supposed to be deterministic
        if parent_info is not None:
            is_deterministic = parent_info.algo_info["child_be_deterministic"]
            child_be_deterministic = is_deterministic
        else:
            is_deterministic = False
            child_be_deterministic = False

        is_deterministic = is_deterministic or force_deterministic
        child_be_deterministic = child_be_deterministic or force_deterministic

        # if child_be_deterministic is true, all children have to use deterministic policies
        # until this node or an active parent gets back control (testing transition)
        child_be_deterministic = child_be_deterministic or np.random.rand() < self._testing_fraction
        new_algo_info = {
                "is_deterministic": is_deterministic,
                "child_be_deterministic": child_be_deterministic
                }
        return new_algo_info
