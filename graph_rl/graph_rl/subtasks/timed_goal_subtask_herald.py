from copy import copy
import os
from typing import Any, Callable, Dict, Mapping, Tuple
from graph_rl.subtasks import subtask

import numpy as np
import numpy.typing as npt

from . import Subtask
from ..spaces import DictSpace, BoxSpace
from ..logging import CSVLogger


class HeraldControlSpec:
    def __init__(self, goal_space_spec, parent_action_space_fields) -> None:
        self._goal_space_spec = goal_space_spec
        self._parent_action_space_fields = parent_action_space_fields

    def calc_intermediate_body_layout(self, init_body_layout, desired_body_layout, high_level_action_horizon, ahead_lookup):
        pass

    def parent_action_space(self, goal_space):
        return DictSpace({
            self._goal_space_spec[key]: goal_space[self._goal_space_spec[key]]
            for key in self._parent_action_space_fields
        })


class HeraldNavigationControlSpec(HeraldControlSpec):
    def __init__(self, goal_space_spec) -> None:
        super().__init__(goal_space_spec, ("position",))

    def calc_intermediate_body_layout(
        self,
        init_body_layout: npt.NDArray[Any],
        desired_body_layout: npt.NDArray[Any],
        high_level_action_horizon: int,
        ahead_lookup: int,
    ) -> Dict[str, Tuple[npt.NDArray[Any]]]:
        """
        Calculate a desired body layout with an use of intermediate points (v2 of the algorithm).

        Args:
            init_body_layout (npt.NDArray[Any]): Initial (first step of the low level action) body layout.
            desired_body_layout (npt.NDArray[Any]): Layout provided by high-level policy i.e. current high-level action.
            high_level_action_horizon (int): Time steps before the end of the current high-level action.
            ahead_lookup (int): Amount of timesteps for which body layout will be calculated.

        Returns:
            npt.NDArray[Any]: Desired body layout will be calculated.
            npt.NDArray[Any]: Desired body velocity.
        """
        n_prime = high_level_action_horizon
        x = init_body_layout[self._goal_space_spec["position"]]
        x_d = desired_body_layout[self._goal_space_spec["position"]]

        lookup_body_layout: npt.NDArray[Any] = x + (x_d - x) / n_prime * ahead_lookup
        return {
            self._goal_space_spec["position"]: lookup_body_layout,
        }

class HeraldNavigationWithVelocityControlSpec(HeraldControlSpec):
    def __init__(self, goal_space_spec) -> None:
        super().__init__(goal_space_spec, ("position",))

    def calc_intermediate_body_layout(
        self,
        init_body_layout: npt.NDArray[Any],
        desired_body_layout: npt.NDArray[Any],
        high_level_action_horizon: int,
        ahead_lookup: int,
    ) -> Dict[str, Tuple[npt.NDArray[Any]]]:
        """
        Calculate a desired body layout with an use of intermediate points (v2 of the algorithm).

        Args:
            init_body_layout (npt.NDArray[Any]): Initial (first step of the low level action) body layout.
            desired_body_layout (npt.NDArray[Any]): Layout provided by high-level policy i.e. current high-level action.
            high_level_action_horizon (int): Time steps before the end of the current high-level action.
            ahead_lookup (int): Amount of timesteps for which body layout will be calculated.

        Returns:
            npt.NDArray[Any]: Desired body layout will be calculated.
            npt.NDArray[Any]: Desired body velocity.
        """
        n_prime = high_level_action_horizon
        x = init_body_layout[self._goal_space_spec["position"]]
        x_d = desired_body_layout[self._goal_space_spec["position"]]

        lookup_body_layout: npt.NDArray[Any] = x + (x_d - x) / n_prime * ahead_lookup
        desired_velocities = (lookup_body_layout - x) / ahead_lookup
        return {
            self._goal_space_spec["position"]: lookup_body_layout,
            self._goal_space_spec["velocity"]: desired_velocities
        }


def default_herald_control_spec_factory(goals_keys_spec: Mapping[str, str]) -> HeraldControlSpec:
    goals_set = set(goals_keys_spec.keys())
    if goals_set == {"position"}:  # variant: navigation
        return HeraldNavigationControlSpec(goals_keys_spec)
    elif goals_set == {"position", "velocity"}:  # variant: navigation
        return HeraldNavigationWithVelocityControlSpec(goals_keys_spec)
    # elif: (other variants)
    else:
        raise ValueError("No herald control spec for fields " + str(goals_set))


class TimedGoalSubtaskHerald(Subtask):
    """Subtask is to achieve a goal at a given point in time."""

    def _ll_reward_sparse(self, obs, ach_time_up, desired_tg, achieved_goal, parent_info):
        # timed goal can only be achieved when the time is up
        if ach_time_up and self.task_spec.goal_achievement_criterion(
            achieved_goal, desired_tg.goal, parent_info
        ):
            reward = 1.0
            achieved = True
        else:

            reward = 0.0
            achieved = False

        return reward, achieved

    def _ll_reward_dist(self, obs, ach_time_up, desired_tg, achieved_goal, parent_info):
        time_passed = self.get_action_time(parent_info.action) - self.task_spec.unconvert_time(desired_tg.delta_t_ach)
        desired = self.control_spec.calc_intermediate_body_layout(
            init_body_layout=parent_info.hl_action_init_obs["observation"],
            desired_body_layout=parent_info.action["goal"],
            high_level_action_horizon=parent_info.hl_action_horizon,
            ahead_lookup=time_passed,
        )

        goal_space = self.parent_action_space["goal"]
        reward = -np.linalg.norm(goal_space.flatten_value(achieved_goal) - goal_space.flatten_value(desired))
        achieved = ach_time_up and self.task_spec.goal_achievement_criterion(
            achieved_goal, desired_tg.goal, parent_info
        )

        return reward, achieved

    def _ll_reward_dist_diff(self, obs, ach_time_up, desired_tg, achieved_goal, parent_info):
        time_passed = self.get_action_time(parent_info.action) - self.task_spec.unconvert_time(desired_tg.delta_t_ach)
        desired = self.control_spec.calc_intermediate_body_layout(
            init_body_layout=parent_info.hl_action_init_obs["observation"],
            desired_body_layout=parent_info.action["goal"],
            high_level_action_horizon=parent_info.hl_action_horizon,
            ahead_lookup=time_passed,
        )
        prev_obs_achieved = self.task_spec.map_to_goal(obs["partial_observation"])

        goal_space = self.parent_action_space["goal"]
        prev_dist = np.linalg.norm(goal_space.flatten_value(prev_obs_achieved) - goal_space.flatten_value(desired))
        dist = np.linalg.norm(goal_space.flatten_value(achieved_goal) - goal_space.flatten_value(desired))

        reward = prev_dist - dist

        achieved = ach_time_up and self.task_spec.goal_achievement_criterion(
            achieved_goal, desired_tg.goal, parent_info
        )

        return reward, achieved

    def __init__(
            self, name, subtask_spec, 
            control_spec_factory: Callable[[Mapping[str, str]], HeraldControlSpec] = default_herald_control_spec_factory):
        """Args:
        subtask_spec (TimedGoalSubtaskSpec): Specification of the timed goal subtask.
        """
        super(TimedGoalSubtaskHerald, self).__init__(name, subtask_spec)

        self._last_n_actions_taken = 0
        self.task_spec = subtask_spec
        
        REWARDS = {
            "default": self._ll_reward_sparse,
            "sparse": self._ll_reward_sparse,
            "dist": self._ll_reward_dist,
            "dist_diff": self._ll_reward_dist_diff
        }

        self.reward = REWARDS[self.task_spec.additional_kwargs.get("reward", "default")]

        self._goal_steps = self.task_spec.additional_kwargs.get("desired_goal_steps", (1, -1))

        self.control_spec = control_spec_factory(self.task_spec._goal_keys_spec)

        obs_dict = {
            "partial_observation": subtask_spec.partial_obs_space,
        }
        obs_dict.update({
            f"desired_goal{i}": subtask_spec.goal_space for i in range(len(self._goal_steps))
        })
        if subtask_spec.include_delta_t_in_obs:
            obs_dict["delta_t_ach"] = BoxSpace(low=[-1.0], high=[1.0])

        self._observation_space = DictSpace(obs_dict)

        self._n_actions_taken = 0

        self.logger_test = None
        self.logger_train = None
        self.logfiles = {}

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def parent_action_space(self):
        space = dict(self.task_spec.parent_action_space)
        space["goal"] = self.control_spec.parent_action_space(space["goal"])
        return DictSpace(space)

    def reset(self):
        self._last_n_actions_taken = 0
        self._n_actions_taken = 0

    def get_action_time(self, action):
        time_conv = action["delta_t_ach"] if self.task_spec.include_delta_t_in_obs else self.task_spec.default_delta_t
        time = self.task_spec.unconvert_time(time_conv)
        return time

    def get_goal_from_obs(self, obs):
        return {key: obs[key] for key in self.task_spec.goal_space.keys()}

    def get_observation(self, env_obs, parent_info, sess_info):
        partial_obs = self.task_spec.map_to_partial_obs(env_obs, parent_info)
        desired_tg = self.task_spec.get_desired_timed_goal(env_obs, parent_info)


        full_obs = {
            "partial_observation": partial_obs,
        }
        for i, step in enumerate(self._goal_steps):
            if step < 0:
                step = self.get_action_time(parent_info.action) + step + 1
            elif step == 0:
                step = self.get_action_time(parent_info.action) + parent_info.step - sess_info.ep_step
            desired = self.control_spec.calc_intermediate_body_layout(
                init_body_layout=parent_info.hl_action_init_obs["observation"],
                desired_body_layout=parent_info.action["goal"],
                high_level_action_horizon=parent_info.hl_action_horizon,
                ahead_lookup=(sess_info.ep_step - parent_info.step) + step,
            )
            full_obs[f"desired_goal{i}"] =  desired
        if self.task_spec.include_delta_t_in_obs:
            # policy sees desired time until achievement relative to current time step
            full_obs["delta_t_ach"] = [
                self.task_spec.convert_time(
                    self.task_spec.unconvert_time(desired_tg.delta_t_ach)
                    - (sess_info.ep_step - parent_info.step)
                )
            ]
        return full_obs

    def check_status(
        self, achieved_goal, desired_tg, obs, action, parent_info, env_info
    ):
        """Assumes that the desired_tg has delta_t attributes in [-1, 1] space and not in units of env steps."""
        ach_time_up = self.task_spec.unconvert_time(desired_tg.delta_t_ach) <= 0.0
        reward, achieved = self.reward(obs, ach_time_up, desired_tg, achieved_goal, parent_info)
        # running out of commitment time
        # NOTE: This is always false if commitment time is NaN
        comm_time_up = self.task_spec.unconvert_time(desired_tg.delta_t_comm) <= 0.0

        # add auxiliary rewards
        if obs is not None and action is not None:
            reward += self.get_aux_rewards(obs, action, env_info.reward)

        return achieved, reward, ach_time_up, comm_time_up

    def _check_status_convenience(
        self, achieved_goal, obs, action, parent_info, sess_info, env_info
    ):
        """Convenience version of check status also does the conversion of delta_t into [-1, 1] interval.

        It assumes that the delta_t in parent_info has not been updated since the emission of the parent_info by
        the parent. Hence, the elapsed time is subtracted from this outdated delta_t."""
        desired_tg = self.get_updated_desired_tg_in_steps(
            env_info.new_obs, parent_info, sess_info
        )
        desired_tg.delta_t_ach = self.task_spec.convert_time(desired_tg.delta_t_ach)
        desired_tg.delta_t_comm = self.task_spec.convert_time(desired_tg.delta_t_comm)

        return self.check_status(
            achieved_goal, desired_tg, obs, action, parent_info, env_info
        )

    def check_interruption(self, env_info, new_subtask_obs, parent_info, sess_info):
        super().check_interruption(env_info, new_subtask_obs, parent_info, sess_info)
        new_env_obs = env_info.new_obs
        new_partial_obs = self.task_spec.map_to_partial_obs(new_env_obs, parent_info)
        achieved_goal = self.task_spec.map_to_goal(new_partial_obs)
        _, _, ach_time_up, comm_time_up = self._check_status_convenience(
            achieved_goal, None, None, parent_info, sess_info, env_info
        )
        return ach_time_up or comm_time_up

    def evaluate_transition(
        self, env_obs, env_info, subtask_trans, parent_info, algo_info, sess_info, child_info
    ):
        new_partial_obs = self.task_spec.map_to_partial_obs(
            env_info.new_obs, parent_info
        )
        achieved_goal = self.task_spec.map_to_goal(new_partial_obs)
        achieved, reward, ach_time_up, comm_time_up = self._check_status_convenience(
            achieved_goal,
            subtask_trans.obs,
            subtask_trans.action,
            parent_info,
            sess_info,
            env_info,
        )

        # subtask ended if subgoal achieved or running out of commitment time
        ended = ach_time_up or comm_time_up

        self._n_actions_taken += 1
        self._last_n_actions_taken = self._n_actions_taken
        if ended or algo_info["interrupted"]:
            n_actions_taken = self._n_actions_taken
            self._n_actions_taken = 0

        # sample delta_t_ach for achieved timed goal from uniform distribution over times that would have run out
        # in this env step
        achieved_timed_goal = self.task_spec.get_achieved_timed_goal_dict(
            achieved_goal=achieved_goal,
            delta_t_ach=sess_info.ep_step - parent_info.step - np.random.rand(),
            parent_info=parent_info,
        )

        # NOTE: The boolean subtask_ended only indicates whether the subtask ended, not that the goal was reached!
        # Whether the goal was reached is encoded in the key "has_achieved" in info.
        info = {
            "has_achieved": achieved,
            "comm_time_up": comm_time_up,
            "ach_time_up": ach_time_up,
            "achieved_generalized_goal": achieved_timed_goal,
        }
        feedback = copy(info)
        # info["delta_t_comm"] = desired_tg.delta_t_comm - (sess_info.ep_step - parent_info.step)
        info["t"] = sess_info.ep_step

        complete_subtask_trans = copy(subtask_trans)
        complete_subtask_trans.reward = reward
        complete_subtask_trans.ended = ended
        complete_subtask_trans.info = info

        # tensorboard logging
        if self._tb_writer is not None and ach_time_up:
            mode = "test" if sess_info.testing else "train"
            self._tb_writer.add_scalar(
                f"{self.name}/{mode}/subgoal_achieved",
                int(achieved),
                sess_info.total_step,
            )
            self._tb_writer.add_scalar(
                f"{self.name}/{mode}/n_actions", n_actions_taken, sess_info.total_step
            )
        # csv logging
        if (
            ach_time_up
            and self.logger_train is not None
            and self.logger_test is not None
        ):
            row_dict = {
                "achieved": int(achieved),
                "n_actions": n_actions_taken,
                "step": sess_info.total_step,
                "time": self.logger_test.time_passed()
                if sess_info.testing
                else self.logger_train.time_passed(),
            }
            if sess_info.testing:
                self.logger_test.log(row_dict)
            elif self.logger_train:
                self.logger_train.log(row_dict)

        return complete_subtask_trans, feedback

    def get_updated_desired_tg_in_steps(self, env_obs, parent_info, sess_info):
        """Returns desired timed goal in global time (i.e. env steps)."""
        desired_tg = self.task_spec.get_desired_timed_goal(env_obs, parent_info)

        elapsed_time = sess_info.ep_step - parent_info.step

        delta_t_ach = desired_tg.delta_t_ach if self.task_spec.include_delta_t_in_obs else self.task_spec.default_delta_t
        delta_t_comm = desired_tg.delta_t_comm if self.task_spec.include_delta_t_in_obs else self.task_spec.default_delta_t

        desired_tg.delta_t_ach = (
            self.task_spec.unconvert_time(delta_t_ach) - elapsed_time
        )
        desired_tg.delta_t_comm = (
            self.task_spec.unconvert_time(delta_t_comm) - elapsed_time
        )

        return desired_tg

    def create_logfiles(self, logdir, append):
        if logdir is not None:
            logfile_test = os.path.join(logdir, self.name + "_test.csv")
            logfile_train = os.path.join(logdir, self.name + "_train.csv")
            self.logger_test = CSVLogger(
                logfile_test, ("achieved", "n_actions", "step", "time"), append
            )
            self.logger_train = CSVLogger(
                logfile_train, ("achieved", "n_actions", "step", "time"), append
            )
            self.logfiles["test"] = logfile_test
            self.logfiles["train"] = logfile_train

    def get_logfiles(self):
        return self.logfiles


class TimedGoalSubtaskHeraldAdditionalState(TimedGoalSubtaskHerald):
    """Subtask is to achieve a goal at a given point in time."""

    def __init__(self, name, subtask_spec):
        """Args:
        subtask_spec (TimedGoalSubtaskSpec): Specification of the timed goal subtask.
        """
        super(TimedGoalSubtaskHeraldAdditionalState, self).__init__(name, subtask_spec)

        self.task_spec = subtask_spec

        additional_state = DictSpace(
            {
                "desired_position": BoxSpace(
                    low=[-1.0, -1.0], high=[1.0, 1.0], dtype=np.float32
                ),
                "desired_velocity": BoxSpace(
                    low=[-1.0, -1.0], high=[1.0, 1.0], dtype=np.float32
                ),
            }
        )

        self._observation_space = DictSpace(
            {
                "partial_observation": subtask_spec.partial_obs_space,
                "additional_state": additional_state,
                "desired_goal": subtask_spec.goal_space,
                "delta_t_ach": BoxSpace(low=[-1.0], high=[1.0]),
            }
        )

        self._n_actions_taken = 0

        self.logger_test = None
        self.logger_train = None
        self.logfiles = {}

    def get_observation(self, env_obs, parent_info, sess_info):
        partial_obs = self.task_spec.map_to_partial_obs(env_obs, parent_info)
        desired_tg = self.task_spec.get_desired_timed_goal(env_obs, parent_info)

        desired = self.control_spec.calc_intermediate_body_layout(
            init_body_layout=parent_info.hl_action_init_obs["observation"]["position"],
            desired_body_layout=parent_info.hl_action_init_obs["desired_goal"],
            high_level_action_horizon=parent_info.hl_action_horizon,
            ahead_lookup=(sess_info.ep_step - parent_info.step) + 1,
        )

        full_obs = {
            "partial_observation": partial_obs,
            "additional_state": {
                "desired": desired,
            },
            "desired_goal": desired_tg.goal,
            # policy sees desired time until achievement relative to current time step
            "delta_t_ach": [
                self.task_spec.convert_time(
                    self.task_spec.unconvert_time(desired_tg.delta_t_ach)
                    - (sess_info.ep_step - parent_info.step)
                )
            ],
        }
        return full_obs
