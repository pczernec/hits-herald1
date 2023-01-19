from typing import Any, Mapping
from gym import Wrapper
from gym.spaces import Box
import numpy as np

from .box_subtask_spec_factory import BoxSubtaskSpecFactory
from graph_rl.spaces import BoxSpace


class HitTargetSubtaskSpecFactory(BoxSubtaskSpecFactory):

    @classmethod
    def get_indices_and_factorization(cls, env, subtask_spec_params, level):
        """Determines observation and goal space and factorization of goal space.
        
        Returns
        partial_obs_indices: Indices of components of "observation" item of
            observation space that are exposed to the level.
        goal_indices: Indices of components of "observation" item of
            observation space that comprise the goal space.
        factorization: List of lists of indices which define the subspaces
            of the goal space in which the Euclidean distance is used.
        """

        obs_space = env.observation_space['observation']

        partial_obs_indices = list(range(obs_space.shape[0]))
        goal_indices = [0, 1]  
        # possibly add velocity to action space, but then it would require a form suitable for comparison (e.g. direction vector)
        factorization = [[i] for i in goal_indices]
        return partial_obs_indices, goal_indices, factorization

    @classmethod
    def get_map_to_env_goal(cls, env):
        """Return map from partial observation to environment goal."""
        def mapping(x):
            return x[:2]
        return mapping

    @classmethod
    def get_map_to_subgoal_and_subgoal_space(cls, env):
        """Return map from partial observation to environment goal and subgoal space."""
        subgoal_space = BoxSpace([-1, -1], [1, 1])
        return cls.get_map_to_env_goal(env), subgoal_space
