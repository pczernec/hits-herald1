from copy import deepcopy

from ..data import FlatTransition
from . import HiTS
from ..subtasks.timed_goal_subtask_specs import TimedGoal


class Herald(HiTS):
    """Herald."""

    def _add_experience_to_flat_algo(
        self, parent_info, deterministic_episode, node_is_sink, sess_info, **kwargs
    ):
        if not sess_info.testing:
            if self._learn_from_deterministic_episodes or not deterministic_episode:

                # add transitions of this episode to replay buffer of flat RL algorithm
                # (by applying hindsight goal and action manipulations)
                for trans_index, tr in enumerate(self._episode_transitions):

                    # unaltered flat transition
                    f_trans_0 = FlatTransition(
                        obs=tr.subtask_tr.obs,
                        action=tr.subtask_tr.action,
                        reward=tr.subtask_tr.reward,
                        new_obs=tr.subtask_tr.new_obs,
                        done=tr.subtask_tr.info["ach_time_up"],
                    )

                    # if the node is a sink, do not attempt to manipulate action
                    # in hindsight and add original transition to replay buffer
                    assert node_is_sink, f"Herald node could be only sink node"
                    f_trans_base = f_trans_0
                    self._add_to_flat_replay_buffer(f_trans_0)

                    # hindsight goal transitions (based on hindsight action
                    # transition or original transition in case of a sink node)
                    achieved_timed_goals = self._sample_achieved_timed_goals(
                        trans_index, parent_info.step
                    )
                    for hindsight_tg in achieved_timed_goals:
                        f_trans_hindsight_goal = deepcopy(f_trans_base)
                        f_trans_hindsight_goal.obs = {
                            "partial_observation": tr.subtask_tr.obs[
                                "partial_observation"
                            ],
                            "additional_state": tr.subtask_tr.obs["additional_state"],
                            "desired_goal": hindsight_tg["desired_goal"],
                            "delta_t_ach": hindsight_tg["delta_t_ach_obs"],
                        }
                        f_trans_hindsight_goal.new_obs = {
                            "partial_observation": tr.subtask_tr.new_obs[
                                "partial_observation"
                            ],
                            "additional_state": tr.subtask_tr.new_obs[
                                "additional_state"
                            ],
                            "desired_goal": hindsight_tg["desired_goal"],
                            "delta_t_ach": hindsight_tg["delta_t_ach_new_obs"],
                        }
                        h_tg = TimedGoal(
                            goal=hindsight_tg["desired_goal"],
                            delta_t_ach=hindsight_tg["delta_t_ach_new_obs"][0],
                            delta_t_comm=hindsight_tg["delta_t_ach_new_obs"][0],
                        )
                        achieved, reward, ach_time_up, _ = self._check_status(
                            achieved_goal=tr.subtask_tr.info[
                                "achieved_generalized_goal"
                            ]["goal"],
                            desired_tg=h_tg,
                            obs=f_trans_hindsight_goal.obs,
                            action=f_trans_hindsight_goal.action,
                            parent_info=parent_info,
                            env_info=tr.env_info,
                        )
                        f_trans_hindsight_goal.done = ach_time_up
                        f_trans_hindsight_goal.reward = reward
                        self._add_to_flat_replay_buffer(f_trans_hindsight_goal)

        self._last_rewards = [
            elem.env_info.reward for elem in self._episode_transitions
        ]
        self._episode_transitions.clear()
