from copy import copy
import os
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

import numpy as np

from . import Subtask
from ..spaces import DictSpace
from ..data import SubtaskTransition
from ..logging import CSVLogger


@dataclass
class ChildInfo:
    ll_n_actions: int
    ll_rewards: List[float]


class ShortestPathSubtask(Subtask):
    """Subtask is to achieve a subgoal with as few actions as possible."""

    def __init__(self, name, subtask_spec):
        """Args:
            subtask_spec (ShortestPathSubtaskSpec): Specification of the shortest path subtask.
        """
        super(ShortestPathSubtask, self).__init__(name, subtask_spec)

        self.task_spec = subtask_spec

        self._observation_space = DictSpace({
            "partial_observation": subtask_spec.partial_obs_space, 
            "desired_goal": subtask_spec.goal_space
            })

        self._n_actions_taken = 0

        self.logger_test = None
        self.logger_train = None
        self.logfiles = {}

        self.reset()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def parent_action_space(self):
        return self.task_spec.parent_action_space

    def reset(self):
        # reset number of actions taken at beginning of episode
        self._n_actions_taken = 0
        self._return = 0

    def get_observation(self, env_obs, parent_info, sess_info):
        partial_obs = self.task_spec.map_to_partial_obs(env_obs, parent_info, 
            sess_info.ep_step)
        desired_goal = self.task_spec.get_desired_goal(env_obs, parent_info)

        full_obs = {
                "partial_observation": partial_obs, 
                "desired_goal": desired_goal, 
                }
        return full_obs

    @property
    def use_original_env_reward(self):
        if hasattr(self.task_spec, '_use_original_reward_of_env'):
            return self.task_spec._use_original_reward_of_env
        else:
            return False

    def check_achievement(self,
                          achieved_goal,
                          desired_goal,
                          obs,
                          action,
                          parent_info,
                          env_info,
                          child_info:Optional[ChildInfo]=None
                          ):

        achieved = self.task_spec.goal_achievement_criterion(achieved_goal, desired_goal, parent_info)

        if self.use_original_env_reward and child_info is not None:
            rewards = [r for r in child_info.ll_rewards]
            if achieved:
                rewards[-1] = self.task_spec._achievement_reward
            if self.task_spec._use_discounted_env_reward:
                reward = np.sum(
                    [reward * self.task_spec._discounting_hl_gamma ** i for i, reward in enumerate(rewards)]).item()
            else:
                reward = np.sum(rewards)
            # Normalize the reward so that we have approximately consistent reward with HiTS
            reward = reward/self.task_spec._reward_normalization_coefficient
        else:
            if achieved:
                reward = 0.
            else:
                reward = -1.
        # add auxiliary rewards
        if obs is not None and action is not None:
            reward += self.get_aux_rewards(obs, action, env_info.reward)
        return achieved, reward

    def check_interruption(self, env_info, new_subtask_obs, parent_info, sess_info):
        super().check_interruption(env_info, new_subtask_obs, parent_info, sess_info)
        new_env_obs = env_info.new_obs
        desired_goal = self.task_spec.get_desired_goal(new_env_obs, parent_info)
        new_partial_obs = self.task_spec.map_to_partial_obs(new_env_obs, parent_info, 
            sess_info.ep_step)
        achieved_goal = self.task_spec.map_to_goal(new_partial_obs)
        achieved, _ = self.check_achievement(achieved_goal, desired_goal, None, None, parent_info, env_info)
        return achieved

    # TODO: refactor this
    def evaluate_transition(self, env_obs, env_info, subtask_trans, parent_info, algo_info, sess_info, child_info:Optional[ChildInfo]=None):
        desired_goal = self.task_spec.get_desired_goal(env_obs, parent_info)
        new_partial_obs = self.task_spec.map_to_partial_obs(env_info.new_obs, parent_info, 
                sess_info.ep_step)
        achieved_goal = self.task_spec.map_to_goal(new_partial_obs)
        achieved, reward = self.check_achievement(achieved_goal, desired_goal,
                                                  subtask_trans.obs, subtask_trans.action, parent_info, env_info,
                                                  child_info=child_info)

        if algo_info.get('include_to_budget', True):
            self._n_actions_taken += 1

        # if constant failure return in wanted, add minus remaining number of actions
        if env_info.done and self.task_spec.constant_failure_return and not achieved:
            reward += (self.task_spec.max_n_actions - self._n_actions_taken)*(0.3*-10. - 1.)  # * (-4)
        self._return += reward

        # subtask ended if subgoal achieved or running out of actions
        if achieved or (self.task_spec.max_n_actions is not None and 
                self._n_actions_taken >= self.task_spec.max_n_actions) or env_info.done:
            n_actions_taken = self._n_actions_taken
            ret = self._return
            self._return = 0
            self._n_actions_taken = 0
            ended = True
        else:
            ended = False

        # NOTE: The boolean ended only indicates whether the subtask ended, not that the goal was reached!
        # Whether the goal was reached is encoded in the key "has_achieved" in info.
        info = {
                "has_achieved": achieved,
                "achieved_generalized_goal": achieved_goal
                }
        feedback = info

        complete_subtask_trans = copy(subtask_trans)
        complete_subtask_trans.reward = reward
        complete_subtask_trans.ended = ended
        complete_subtask_trans.info = info

        # tensorboard logging
        if self._tb_writer is not None:
            if ended == True:
                mode = "test" if sess_info.testing else "train"
                self._tb_writer.add_scalar(f"{self.name}/{mode}/subgoal_achieved", int(achieved), sess_info.total_step)
                self._tb_writer.add_scalar(f"{self.name}/{mode}/n_actions", n_actions_taken, sess_info.total_step)
                self._tb_writer.add_scalar(f"{self.name}/{mode}/return", ret, sess_info.total_step)
        # csv logging
        if ended and self.logger_train is not None and self.logger_test is not None:
            row_dict = {
                    "achieved": int(achieved), 
                    "n_actions": n_actions_taken, 
                    "step": sess_info.total_step, 
                    "time": self.logger_test.time_passed() if sess_info.testing else self.logger_train.time_passed()
                    }
            if sess_info.testing:
                self.logger_test.log(row_dict)
            elif self.logger_train:
                self.logger_train.log(row_dict)

        return complete_subtask_trans, feedback

    def create_logfiles(self, logdir, append = False):
        if logdir is not None:
            logfile_test = os.path.join(logdir, self.name  + "_test.csv")
            logfile_train = os.path.join(logdir, self.name  + "_train.csv")
            self.logger_test = CSVLogger(logfile_test, ("achieved", "n_actions", "step", "time"), append)
            self.logger_train = CSVLogger(logfile_train, ("achieved", "n_actions", "step", "time"), append)
            self.logfiles["test"] = logfile_test
            self.logfiles["train"] = logfile_train

    def get_logfiles(self):
        return self.logfiles
