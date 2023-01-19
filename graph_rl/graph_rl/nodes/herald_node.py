import copy
from dataclasses import dataclass

import numpy as np
import torch
import wandb
from typing import Callable, Tuple, Any, Dict, Optional

from dyn_rl_benchmarks.envs.platforms_env import PlatformsEnv
from scripts.helpers.utils import ExponentialAverageCalculator, RunningStd
from . import Node
from ..subtasks import DictInfoHidingTolTGSubtaskSpec, TimedGoalSubtask
from ..algorithms import Herald, HiTS
from ..subtasks.timed_goal_subtask_herald import (
    TimedGoalSubtaskHerald,
    TimedGoalSubtaskHeraldAdditionalState,
)
Q_ACCUMULATON_START_ENV_TIMESTEPS = 0.05
Q_INTERRUPTION_START_ENV_TIMESTEPS = 0.15


@dataclass
class InterruptionHelper:
    envs_timesteps = {
        "Reacher": 300_000,
        "Ant": 1_200_000,
        "Pendulum": 150_000,
        "Ball": 300_000,
        "Platforms": 5_000_000,
        "Drawbridge": 500_000,
        "Tennis": 20_000_000,
    }

    envs_q_statistics_accumulation_start = {
        key: int(elem * Q_ACCUMULATON_START_ENV_TIMESTEPS)
        for key, elem in envs_timesteps.items()
    }
    envs_interruption_start = {
        key: int(elem * Q_INTERRUPTION_START_ENV_TIMESTEPS)
        for key, elem in envs_timesteps.items()
    }
    def __init__(self, env_class_name:str, force_interruption_start_at: Optional[int]=None):
        if force_interruption_start_at is not None:
            self.q_statistics_accumulation_start = int(force_interruption_start_at/2)
            self.interruption_start = force_interruption_start_at

        else:
            key = [key for key in self.envs_timesteps.keys() if key in env_class_name][0]
            self.q_statistics_accumulation_start = self.envs_q_statistics_accumulation_start[key]
            self.interruption_start = self.envs_interruption_start[key]

        print(f"Using timesteps:\n"
              f"q_statistics_accumulation_start={self.q_statistics_accumulation_start}\ninterruption_start={self.interruption_start}\n"
              f"force_interruption_start_at is {force_interruption_start_at}")


class HeraldNode(Node):
    def __init__(
        self,
        name,
        parents,
        subtask_spec,
        HiTS_kwargs,
        HERALD_kwargs,
        env=None,
        node_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if node_kwargs is None:
            node_kwargs = {}

        self._set_env_logging(env)

        # For action interruption
        self._interruption_function: Callable[..., Tuple[bool, Dict[str, Any]]]
        self._interruption_type = HERALD_kwargs.get(
            "interruption_type", "q_function_exponential"
        )
        self.set_interruption_function()

        # Different setups for different interruptions
        self._goal_distance = 0
        self._goal_distance_threshold = HERALD_kwargs.get(
            "goal_distance_threshold", 0.4
        )
        self._q_value_ratio_threshold = HERALD_kwargs.get(
            "q_value_ratio_threshold", 0.5
        )
        self._goal_distance_relative = HERALD_kwargs.get("goal_distance_relative", 0.4)

        self._goal_distance_activate_steps = HERALD_kwargs.get("goal_distance_activate_steps", None)

        self.interruption_helper = InterruptionHelper(env_class_name=env.__repr__(),
                                                      force_interruption_start_at=self._goal_distance_activate_steps)
        self._exp_avg_alpha = HERALD_kwargs.get("exp_avg_alpha", 0.999)
        self._exp_q_val_avg_calculator = ExponentialAverageCalculator(self._exp_avg_alpha,
                                                                      statistics_accumulation_start=self.interruption_helper.q_statistics_accumulation_start)
        self._exp_q_val_std_calculator = RunningStd(self._exp_avg_alpha,
                                                    statistics_accumulation_start=self.interruption_helper.q_statistics_accumulation_start)

        self._q_value_std_multiplyer = HERALD_kwargs.get("q_value_std_multiplyer", 1)
        self._goal_distance_relative = HERALD_kwargs.get(
            "goal_distance_relative", False
        )
        self._q_val_thre_normalization = HERALD_kwargs.get(
            "q_val_thre_normalization", 0.1
        )

        self._timed_goal_subtask_type = HERALD_kwargs.get("timed_goal_subtask", "default")
        if self._timed_goal_subtask_type == "herald_additional_state":
            timed_subgoal_cls = TimedGoalSubtaskHeraldAdditionalState
            algo_cls = Herald
        elif self._timed_goal_subtask_type == "herald":
            timed_subgoal_cls = TimedGoalSubtaskHerald
            algo_cls = HiTS
        elif self._timed_goal_subtask_type == "hits" or self._timed_goal_subtask_type == "default":
            timed_subgoal_cls = TimedGoalSubtask
            algo_cls = HiTS
        else:
            raise Exception(f"No ")

        # create subtask
        subtask = timed_subgoal_cls(name + "_subtask", subtask_spec)

        assert HiTS_kwargs.get('n_hindsight_goals', 0) == 0 or (self._timed_goal_subtask_type=="hits" or self._timed_goal_subtask_type == "default"),\
            f"Herald no longer supports hinsight goals for _timed_goal_subtask_type={self._timed_goal_subtask_type}. Set n_hindsight_goals to 0."

        # create algorithm
        algorithm = algo_cls(
            name=name + "_algorithm",
            check_status=subtask.check_status,
            convert_time=subtask.task_spec.convert_time,
            delta_t_min=subtask.task_spec._delta_t_min,
            use_normal_trans_for_testing=False,
            use_testing_transitions=True,
            learn_from_deterministic_episodes=True,
            **HiTS_kwargs,
        )

        # policy creation is done via algorithm because it does the sampling part
        # of generating an action
        policy_class = algorithm.get_policy

        self.current_timed_goal = None
        self.current_goal_tol = None

        super(HeraldNode, self).__init__(
            name, policy_class, subtask, algorithm, parents, **node_kwargs
        )

        self._add_experience_to_parent_every_step = HERALD_kwargs.get('add_experience_to_parent_every_step', False)
        self.debug = HERALD_kwargs.get("debug", False)

    def __repr__(self):
        return f"{self.__class__.__name__}\n\t{super(HeraldNode, self).__repr__()}"

    def check_parent_interruption(
        self, env_info, child_feedback, sess_info, check_this_node=False
    ):
        parent_info = self._parent_info[-1]
        self.current_timed_goal = self.subtask.get_updated_desired_tg_in_steps(
            env_info.new_obs, parent_info, sess_info
        )
        if isinstance(self.subtask.task_spec, DictInfoHidingTolTGSubtaskSpec):
            self.current_goal_tol = parent_info.action["goal_tol"]
        elif hasattr(self.subtask.task_spec, "_goal_achievement_threshold"):
            self.current_goal_tol = self.subtask.task_spec._goal_achievement_threshold
        else:
            self.current_goal_tol = None
        return super(HeraldNode, self).check_parent_interruption(
            env_info, child_feedback, sess_info, check_this_node
        )

    def get_parameters(self):
        parameters = self.algorithm.get_parameters()

        parameters["additional"] = {"avg_calc": {"y_t": self._exp_q_val_avg_calculator.y_t},
                                    "std_calc": {"y_t": self._exp_q_val_std_calculator.y_t,
                                                 "Ex": self._exp_q_val_std_calculator.Ex,
                                                 "Ex2": self._exp_q_val_std_calculator.Ex2}}
        return parameters

    def load_parameters(self, params):
        if "additional" in params:
            additional_params = params.pop("additional")
            self._exp_q_val_avg_calculator.load(float(additional_params["avg_calc"]["y_t"]))
            self._exp_q_val_std_calculator.load(float(additional_params["std_calc"]["y_t"]),
                                                float(additional_params["std_calc"]["Ex"]),
                                                float(additional_params["std_calc"]["Ex2"]))
        super().load_parameters(params)

    @torch.no_grad()
    def if_interrupt(self, sess_info):
        is_interrupted, dict_to_log = self._interruption_function()

        # Should it be time of the HL action or current time?
        if self._parents[0]._sess_info[-1].total_step > self.interruption_helper.interruption_start or sess_info.rendering:
            self._parents[0].episode_info.interruptions += int(is_interrupted)
            if is_interrupted:
                self._parents[0].episode_info.actions_interrupted_len_actual.append(len(self.algorithm._episode_transitions))
                self._parents[0].episode_info.actions_interrupted_len_predicted.append(
                    self.subtask.task_spec.unconvert_time(self._parent_info[-1].action['delta_t_ach']).item())
                self._parents[0].episode_info.interruption_times.append(sess_info.ep_step)
        else:
            is_interrupted = False

        if self._should_log():
            try:
                wandb.log(
                    dict_to_log,
                )
            except:
                print(f"Probably you are just rendering")
        return is_interrupted

    def set_interruption_function(self) -> None:
        if self._interruption_type == "relative_policy_threshold":
            self._interruption_function = (
                self.calc_subgoals_relative_distance_policy_interruption
            )
        elif self._interruption_type == "q_function_exponential":
            self._interruption_function = (
                self.calc_q_func_value_interruption_exponential_average
            )
        elif self._interruption_type == "q_normalized":
            self._interruption_function = self.calc_q_func_value_interruption_state_normalized
        elif self._interruption_type == "dummy":
            self._interruption_function = lambda *args, **kwargs: (
                False,
                {"q_values/interrupt": 0},
            )
        elif self._interruption_type == "frequent_dummy":
            self._interruption_function = lambda *args, **kwargs: (
                True,
                {"q_values/interrupt": 1},
            )
        else:
            raise Exception(
                f"Not implemented interruption policy: {self._interruption_type}"
            )

    def _flatten_action(self, goal, ref):
        if isinstance(goal, dict):
            return np.concatenate([goal[key] for key in ref.keys()])
        else:
            return goal

    def _flatten_obs(self, obs, ref):
        if isinstance(obs['observation'], dict):
            return np.concatenate([obs['observation'][key] for key in ref.keys()])
        else:
            return self.subtask.task_spec.map_to_goal(obs['observation'])

    def _calc_action_difference_cos_and_vel(self, action_init, action_init_time, action_now, action_now_time):
        goal_init_flattened = self._flatten_action(action_init['goal'], action_init['goal'])
        goal_new_flattened = self._flatten_action(action_now['goal'], action_init['goal'])
        obs_current = self._env_obs[-1]
        goal_achieved = self._flatten_obs(obs_current, action_init['goal'])

        vec_init = goal_init_flattened - goal_achieved
        vec_new = goal_new_flattened - goal_achieved

        scaled_vec_init = vec_init / action_init_time
        scaled_vec_now = vec_new / action_now_time

        diff = np.linalg.norm(scaled_vec_init - scaled_vec_now)
        if self._goal_distance_relative:
            diff = (
                2 * diff
                / (np.linalg.norm(scaled_vec_now) + np.linalg.norm(scaled_vec_init))
            )
        return diff

    def _calc_action_difference(self, action_init, action_init_time, action_now, action_now_time):

        desired_goal_init = self.subtask.control_spec.calc_intermediate_body_layout(
            init_body_layout=self._parents[0]._env_obs[0]["observation"],
            desired_body_layout=action_init["goal"],
            high_level_action_horizon=action_init_time,
            ahead_lookup=self.subtask._n_actions_taken + 1,
        )

        desired_goal_new = self.subtask.control_spec.calc_intermediate_body_layout(
            self._env_obs[0]["observation"],  # Here we take current position
            desired_body_layout=action_now["goal"],
            high_level_action_horizon=action_now_time,
            ahead_lookup=1,
        )

        # both below lines use the same keys collection
        desired_goal_init_flattened = self._flatten_action(desired_goal_init, desired_goal_init)
        desired_goal_new_flattened = self._flatten_action(desired_goal_new, desired_goal_init)

        distance_between_next_steps = np.linalg.norm(
            desired_goal_init_flattened - desired_goal_new_flattened
        )

        if self._goal_distance_relative:
            return (
                2
                * distance_between_next_steps
                / (
                    np.linalg.norm(desired_goal_init_flattened)
                    + np.linalg.norm(desired_goal_new_flattened)
                )
            )
        else:
            return distance_between_next_steps


    def calc_subgoals_relative_distance_policy_interruption(
        self,
    ) -> Tuple[bool, Dict[str, Any]]:
        self._get_hl_state()
        hl_policy_for_inference = self._parents[0].policy

        assert len(self._parents[0]._env_obs) == 1

        action_init_hl_action = self._get_predicted_hl_action(
            hl_policy_for_inference, self._parents[0]._env_obs[-1], self._parents[0]._sess_info[-1]
        )
        action_timestep_now = self._get_predicted_hl_action(
            hl_policy_for_inference, self._env_obs[-1], self._sess_info[-1]
        )
        action_init_hl_action_time = self.subtask.get_action_time(action_init_hl_action)
        action_timestep_now_time = self.subtask.get_action_time(action_timestep_now)

        if self._timed_goal_subtask_type == "herald":
            distance_between_next_steps = self._calc_action_difference(
                action_init_hl_action, action_init_hl_action_time,
                action_timestep_now, action_timestep_now_time
            )
        else:
            distance_between_next_steps = self._calc_action_difference_cos_and_vel(
                action_init_hl_action, action_init_hl_action_time,
                action_timestep_now, action_timestep_now_time
            )

        interrupt = distance_between_next_steps > self._goal_distance_threshold

        # Generic dict to log
        dict_to_log = self._env_specific_logging_dict(
            action_init_hl_action=action_init_hl_action,
            action_timestep_now=action_timestep_now,
        )
        dict_to_log.update(
            self.generic_logging(self._sess_info[-1].total_step, interrupt)
        )
        # Function specific dict to log
        dict_to_log.update(
            {
                "hl_action_goal/next_step_distance": distance_between_next_steps,
                "q_values/interrupt": 1 if interrupt else 0,
            }
        )
        self._set_hl_state(critics_for_inference=None, hl_policy_for_inference=hl_policy_for_inference)
        return interrupt, dict_to_log

    def calc_q_func_value_interruption_exponential_average(
        self,
    ) -> Tuple[bool, Dict[str, Any]]:
        self._get_hl_state()
        hl_policy_for_inference = self._parents[0].policy
        critics_for_inference = [
            (critic).train(False)
            for critic in self._parents[0].algorithm._model.critics
        ]

        action_init_hl = self._get_predicted_hl_action(
            hl_policy_for_inference,
            self._parents[0]._env_obs[-1], self._parents[0]._sess_info[-1],
        )
        action_timestep_now = self._get_predicted_hl_action(
            hl_policy_for_inference, self._env_obs[-1], self._sess_info[-1]
        )

        # Time passed from HL action initialization
        init_action_timesteps = self.subtask.task_spec.unconvert_time(action_init_hl['delta_t_ach'])
        time_passed_since_hl_init = self._act_time[-1] - self._parents[0]._act_time[-1]
        time_till_action_change = init_action_timesteps-time_passed_since_hl_init
        time_till_action_change_converted = self.subtask.task_spec.convert_time(time_till_action_change)

        # Modify action time delta_t_ach
        action_init_hl_action_modified = copy.deepcopy(action_init_hl)
        action_init_hl_action_modified['delta_t_ach'] = time_till_action_change_converted

        # Here we take current observation
        q_value_init = self._q_value_now_predicted_hl_action(
            action_init_hl_action_modified, critics_for_inference
        )
        q_value_now = self._q_value_now_predicted_hl_action(
            action_timestep_now, critics_for_inference
        )

        q_value_average = self._exp_q_val_avg_calculator(
            q_value_now,
            self._sess_info[-1].total_step,
            learn=self.algorithm._episode_transitions[-1].sess_info.learn  # TODO: take care of it
        )
        q_value_std = self._exp_q_val_std_calculator(
            q_value_now,
            q_value_average,
            self._sess_info[-1].total_step,
            learn=self.algorithm._episode_transitions[-1].sess_info.learn
        )

        interrupt = (
            q_value_now > q_value_init + q_value_std * self._q_value_std_multiplyer
        )

        dict_to_log = self._env_specific_logging_dict(
            action_init_hl_action=action_init_hl,
            action_timestep_now=action_timestep_now,
        )
        dict_to_log.update(
            self.generic_logging(self._sess_info[-1].total_step, interrupt)
        )
        dict_to_log.update(
            {
                "q_values/q_value_init": q_value_init,
                "q_values/q_value_now": q_value_now,
                "q_values/q_value_average": q_value_average,
                "q_values/q_value_std": q_value_std,
                "q_values/interrupt": 1 if interrupt and
                                           dict_to_log['global_step'] >= self.interruption_helper.interruption_start else 0,
            }
        )
        self._set_hl_state(critics_for_inference, hl_policy_for_inference)
        return interrupt, dict_to_log

    def calc_q_func_value_interruption_state_normalized(
        self,
    ) -> Tuple[bool, Dict[str, Any]]:
        self._get_hl_state()
        hl_policy_for_inference = self._parents[0].policy
        critics_for_inference = [
            (critic).train(False)
            for critic in self._parents[0].algorithm._model.critics
        ]

        action_init_best = self._get_predicted_hl_action(
            hl_policy_for_inference,
            self._parents[0]._env_obs[-1], self._parents[0]._sess_info[-1],
        )

        action_now_best = self._get_predicted_hl_action(
            hl_policy_for_inference, self._env_obs[-1], self._sess_info[-1]
        )

        # Time passed from HL action initialization
        init_action_timesteps = self.subtask.task_spec.unconvert_time(action_init_best['delta_t_ach'])
        time_passed_since_hl_init = self._act_time[-1] - self._parents[0]._act_time[-1]
        time_till_action_change = init_action_timesteps - time_passed_since_hl_init
        time_till_action_change_converted = self.subtask.task_spec.convert_time(time_till_action_change)

        # Modify action time delta_t_ach
        action_init_hl_action_modified = copy.deepcopy(action_init_best)
        action_init_hl_action_modified['delta_t_ach'] = time_till_action_change_converted

        q_value_now_best = self._q_value_now_predicted_hl_action(
            action_now_best, critics_for_inference
        )
        q_value_init_best = self._q_value_now_predicted_hl_action(
            action_init_hl_action_modified, critics_for_inference
        )

        difference_now = q_value_now_best - q_value_init_best

        interrupt = difference_now > 0 and difference_now > self._q_val_thre_normalization*np.abs(q_value_init_best)

        dict_to_log = self._env_specific_logging_dict(
            action_init_hl_action=action_init_best,
            action_timestep_now=action_now_best,
        )
        dict_to_log.update(
            self.generic_logging(self._sess_info[-1].total_step, interrupt)
        )
        dict_to_log.update(
            {
                "q_values/q_value_init": q_value_init_best,
                "q_values/q_value_now": q_value_now_best,
                "q_values/interrupt": 1 if interrupt else 0,
            }
        )
        self._set_hl_state(critics_for_inference, hl_policy_for_inference)
        return interrupt, dict_to_log

    def _set_hl_state(self, critics_for_inference=None, hl_policy_for_inference=None):
        if hl_policy_for_inference is not None:
            hl_policy_for_inference._ts_policy.train(self._hl_actor_is_training)
        if critics_for_inference is not None:
            for critic, critic_is_training in zip(critics_for_inference, self._hl_critics_training):
                critic.train(critic_is_training)

    def _get_hl_state(self):
        self._hl_actor_is_training = self._parents[0].policy._ts_policy.training
        self._hl_critics_training = [
            critic.training
            for critic in self._parents[0].algorithm._model.critics
        ]

    def _get_predicted_hl_action(self, hl_policy_for_inference, env_obs, sess_info):
        """HL action created right now"""
        subtask_obs = self._parents[0].subtask.get_observation(
            env_obs,
            None,
            sess_info,
        )  # NOTE: None because there are no parents for hl policy
        algo_info = self._parents[0].algorithm.get_algo_info(
            env_obs, None, force_deterministic=True
        )
        _, action = hl_policy_for_inference(
            subtask_obs, algo_info, testing=True
        )
        return action

    def _obs_for_hl_based_on_ll(self):
        obs_to_merge_ll_hl = {}
        obs_to_merge_ll_hl["desired_goal"] = self._parents[0]._env_obs[-1]["desired_goal"]
        obs_to_merge_ll_hl["partial_observation"] = self._env_obs[-1]["observation"]
        return obs_to_merge_ll_hl

    def _q_value_now_predicted_hl_action(
        self, action, critics_for_inference
    ):
        """Q value prediction of now designated HL action"""
        obs_now = self._obs_for_hl_based_on_ll()
        obs_now = self._parents[0].algorithm._observation_space.flatten_value(obs_now)
        action_flattened = self._parents[0].algorithm._action_space.flatten_value(
            action
        )
        q_value_now = self._calc_q_value(action_flattened, critics_for_inference, obs_now)
        return q_value_now

    def _set_env_logging(self, env):
        if isinstance(env.env, PlatformsEnv):
            self._env_specific_logging_dict = PlatformsEnv.create_generic_wandb_dict
        else:
            print(f"No logging for env: {env.env}")
            self._env_specific_logging_dict = lambda *args, **kwargs: {}

    @staticmethod
    def _calc_q_value(action, critics_for_inference, observation):
        q_value_init = np.mean(
            [
                critic(observation[None], action[None]).cpu().detach().numpy()
                for critic in critics_for_inference
            ]
        ).item()
        return q_value_init

    @staticmethod
    def generic_logging(
        total_step,
        interrupt,
    ):
        return {
            "q_values/interrupt": 1 if interrupt else 0,
            "global_step": total_step,
        }
