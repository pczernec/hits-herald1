import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
from Box2D import (
    b2PolygonShape,
    b2World,
    b2FixtureDef,
    b2CircleShape,
    b2Vec2,
)

from ..utils.graphics_utils import ArrowConfig, get_default_subgoal_colors


class _MetaFreezer(ABC):
    use_freeze: bool
    is_frozen: bool
    number_of_freezes: int
    freeze_probability: float
    max_number_of_freezes: int
    time_since_frozen: float
    total_frozen_time: float  # used in platform's position calculation
    max_freeze_time: float

    @abstractmethod
    def update(self, agent_pos: b2Vec2, delta_t: float):
        pass

    def reset(self):
        if self.use_freeze:
            self.unfreeze()
            self.total_frozen_time = 0.
            self.number_of_freezes = 0.

    def exec_freeze_logic(self, delta_t):
        # -------------------- Freeze logic -----------------------
        # Check whether to freeze
        if self.number_of_freezes < self.max_number_of_freezes \
            and self.time_since_frozen == 0. \
            and np.random.binomial(n=1, p=self.freeze_probability) == 1:
            self.number_of_freezes += 1
            self.is_frozen = True
        # Check freeze end time
        if self.is_frozen and self.time_since_frozen >= self.max_freeze_time:
            self.unfreeze()
        # Update freeze time
        if self.is_frozen and self.time_since_frozen >= 0.:
            self.time_since_frozen += delta_t
        if self.is_frozen:
            self.total_frozen_time += delta_t
        # ---------------------------------------------------------

    def unfreeze(self):
        self.is_frozen = False
        self.time_since_frozen = 0.


class PlatformsEnv(gym.GoalEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    class _FreezeTrigger(_MetaFreezer):

        def __init__(
            self,
            use_freeze: bool = False,
            third_platform: bool = False,
            freeze_probability: float = 0.005,
            max_number_of_freezes: int = 2,
            max_freeze_time: int = 50,
        ) -> None:
            self.use_freeze = use_freeze
            self.third_platform = third_platform
            self.is_frozen = False
            self.number_of_freezes = 0
            self.freeze_probability = freeze_probability
            self.max_number_of_freezes = max_number_of_freezes
            self.time_since_frozen = 0.
            self.total_frozen_time = 0.
            self.max_freeze_time = max_freeze_time

        def update(self, _, delta_t):
            if self.use_freeze:
                self.exec_freeze_logic(delta_t)

        def get_active_time(self):
            if self.use_freeze:
                return self.total_frozen_time
            else:
                return 0

    class _Trigger(_MetaFreezer):

        def __init__(
            self,
            position: Optional[float] = None,
            width: Optional[float] = None,
            height: Optional[float] = None,
            max_act_time: Optional[float] = None,
            use_freeze: bool = False,
            third_platform: bool = False,
            freeze_probability: float = 0.005,
            max_number_of_freezes: int = 2,
            max_freeze_time: int = 50,
        ):
            self.pos = position
            self.width = width
            self.height = height
            self.max_act_time = max_act_time
            self.is_active = False
            self.time_since_act = 0.

            self.use_freeze = use_freeze
            self.third_platform = third_platform
            self.is_frozen = False
            self.number_of_freezes = 0
            self.freeze_probability = freeze_probability
            self.max_number_of_freezes = max_number_of_freezes
            self.time_since_frozen = 0.
            self.total_frozen_time = 0.
            self.max_freeze_time = max_freeze_time

        def update(self, agent_pos, delta_t):
            if self.is_active:
                self.time_since_act += delta_t
                if self.time_since_act >= self.max_act_time:
                    self.is_active = False

            if self.is_active and self.use_freeze:
                self.exec_freeze_logic(delta_t)

                if self.is_frozen:
                    self.max_act_time += delta_t  # allow a platform to return to its starting position

            if not self.is_active:
                if self.is_frozen:
                    self.unfreeze()

                if (self.pos[0] - 0.5 * self.width < agent_pos[0] < self.pos[0] + 0.5 * self.width) \
                    and (self.pos[1] - 0.5 * self.height < agent_pos[1] < self.pos[1] + 0.5 * self.height):
                    self.is_active = True
                    self.time_since_act = 0

        def reset(self):
            super().reset()
            self.is_active = False
            self.time_since_act = 0.

        def get_active_time(self):
            if self.use_freeze:
                return self.time_since_act - self.total_frozen_time
            else:
                return self.time_since_act

    def __init__(
        self,
        max_episode_length=500,
        subgoal_radius=0.05,
        use_simpler_goal=False,
        use_freeze: Tuple[bool, bool] = (False, False),
        third_platform: Tuple[bool] = (False),
        freeze_probability: float = 0.005,
        max_number_of_freezes: int = 2,
        max_freeze_time: int = 50,
    ):
        super().__init__()

        print(f'Platforms noise setup {str(use_freeze)}')

        self.max_episode_length = max_episode_length
        self.current_step = 0
        self.level_width = 2.0

        # safety distance from boundaries of interval [-1., 1.]
        self._eps = 1e-1

        self.window = None
        self.window_width = 800
        self.window_height = 600
        self.background_color = (1.0, 1.0, 1.0, 1.0)

        self._vector_width = 0.02
        self._arrow_head_size = 0.05
        self._velocity_color = (0.0, 0.0, 0.0)
        self._velocity_scale = 0.2

        self.static_color = (0.4, 0.4, 0.4)
        self.platform_color = (0.702, 0.612, 0.51)

        self.agent_radius = 0.05
        self.agent_color = (0.0, 0.0, 0.0)
        self.line_color = (0.5, 0.5, 0.5)

        self._agent_ac = ArrowConfig(
            self._velocity_scale,
            self._vector_width,
            self._arrow_head_size,
            self.line_color,
        )

        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = []
        self._subgoal_colors = get_default_subgoal_colors()
        self._subgoal_acs = [
            ArrowConfig(self._velocity_scale, self._vector_width, self._arrow_head_size, color)
            for color in self._subgoal_colors
        ]
        self.subgoal_radius = float(subgoal_radius) * self.level_width * 0.5

        self.goal_radius = 0.05
        self.goal_color = (0.0, 0.0, 0.0)

        self.static_rects = []
        self.kinematic_rects = []

        # set up physics in Box2D
        self.world = b2World(gravity=(0.0, -9.8))
        self.vel_iters, self.pos_iters = 10, 8
        self.time_step = 1.0 / 60.0

        # static elements of level
        ground_level = -0.6
        # keep ball from falling out of play area
        self._create_static_b2_rect(-0.0, -0.5 * self.level_width - 0.5, self.level_width, 1.0)
        self._create_static_b2_rect(-0.25, ground_level - 0.35, 1.5, 0.7)  # ground
        self._create_static_b2_rect(-0.5 * self.level_width - 0.5, 0.0, 1.0, 10)  # left wall
        self._create_static_b2_rect(0.5 * self.level_width + 0.5, 0.0, 1.0, 10)  # right wall

        level_1 = -0.1
        diff1 = level_1 - ground_level
        self.level_2 = level_1 + diff1
        self.level_1 = level_1
        self.level_3 = self.level_2 + diff1
        level_thickness = 0.1
        self._create_static_b2_rect(-0.5, self.level_2 - 0.5 * level_thickness, 1.0, level_thickness)  # level 1

        #self._create_static_b2_rect(-1.0, self.level_2 - 0.5 * level_thickness+0.2, 1.0, level_thickness)  # level 1

        self.agent_initial_position = np.array((-0.8, ground_level + self.agent_radius))

        # triggers
        self.active_trigger_color = (0.2, 0.8, 0.2)
        self.inactive_trigger_color = (0.8, 0.2, 0.2)
        self.frozen_trigger_color = (0.2, 0.2, 0.8)
        self.triggers: List[_MetaFreezer] = []
        omega2 = 2.0
        self.triggers.append(
            self._FreezeTrigger(
                use_freeze=use_freeze[0],
                freeze_probability=freeze_probability,
                max_number_of_freezes=max_number_of_freezes,
                max_freeze_time=max_freeze_time,
            ))
        self.triggers.append(
            self._Trigger(
                position=[-0.0, ground_level + 0.05],
                width=0.1,
                height=0.1,
                max_act_time=2.0 * np.pi / self.time_step / omega2,
                use_freeze=use_freeze[1],
                freeze_probability=freeze_probability,
                max_number_of_freezes=max_number_of_freezes,
                max_freeze_time=max_freeze_time,
            ))

        self.third_platform = third_platform

        # dynamic elements (platforms)
        omega1 = 3
        # continually moving platform
        phi1 = -0.8 * np.pi

        def pos_func_1(t: float) -> Tuple[float, float]:
            x = 0.75
            y = 0.5 * (ground_level + level_1) \
                - 0.5 * level_thickness \
                + 0.5 * diff1 * np.sin(omega1 * (t - self.triggers[0].get_active_time() * self.time_step) + phi1)

            return x, y

        def vel_func_1(t: float) -> Tuple[float, float]:
            x = 0.
            y = omega1 * 0.5 * diff1 * np.cos(omega1 * (t - self.triggers[0].get_active_time() * self.time_step) + phi1)

            return x, y

        self._create_kinematic_b2_rect(
            pos_func=pos_func_1,
            vel_func=vel_func_1,
            width=0.5,
            height=level_thickness,
            color=self.platform_color,
        )

        # triggered platform
        phi2 = 0.

        def pos_func_2(t: float) -> Tuple[float, float]:
            x = 0.25
            y = level_1 \
                + 0.5 * diff1 \
                - 0.5 * level_thickness \
                + 0.5 * diff1 * np.cos(omega2 * self.triggers[1].get_active_time() * self.time_step + phi2)

            return x, y

        def vel_func_2(t: float) -> Tuple[float, float]:
            x = 0.
            y = 0.
            if self.triggers[1].is_active and not self.triggers[1].is_frozen:
                y = -omega1 * 0.5 * diff1 * np.sin(omega2 * self.triggers[1].get_active_time() * self.time_step + phi2)

            return x, y

        self._create_kinematic_b2_rect(
            pos_func=pos_func_2,
            vel_func=vel_func_2,
            width=0.5,
            height=level_thickness,
            color=self.inactive_trigger_color,
        )

        self._create_static_b2_rect(-0.75, self.level_2 - 0.5 * level_thickness+diff1, 0.5, level_thickness)

        def pos_func_3(t: float) -> Tuple[float, float]:
            x = -0.25
            y = self.level_2 \
                - 0.5 * level_thickness
            if third_platform == [True]:
                y = y + 0.5 * diff1 + 0.5 * diff1 * np.sin(omega1 * (t - self.triggers[0].get_active_time() * self.time_step) + phi1)

            return x, y

        def vel_func_3(t: float) -> Tuple[float, float]:
            x = 0.
            y = 0.
            if third_platform == [True]:
                 y = omega1 * 0.5 * diff1 * np.cos(omega1 * (t - self.triggers[0].get_active_time() * self.time_step) + phi1)

            return x, y

        self._create_kinematic_b2_rect(
            pos_func=pos_func_3,
            vel_func=vel_func_3,
            width=0.5,
            height=level_thickness,
            color=self.platform_color,
        )

        # ball/agent
        self.max_force = 1e-1
        # self.max_vel_comp = 1e-1
        self.max_vel_comp = 1
        self.max_ang_vel = (2 * np.pi * self.max_vel_comp / (2 * np.pi * self.agent_radius))
        ball_fixture = b2FixtureDef(
            shape=b2CircleShape(pos=(0.0, 0.0), radius=self.agent_radius),
            density=1.0,
            restitution=0.4,
        )
        self.ball = self.world.CreateDynamicBody(position=self.agent_initial_position, fixtures=ball_fixture)

        # define spaces
        desired_goal_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        achieved_goal_space = desired_goal_space
        obs_space_dict = {
            "position": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "velocity": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "ang_vel": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        }
        obs_platforms_dict = {
            "platform{}".format(i): spaces.Box(
                low=np.array([-1.0, -1.0, -np.inf, -np.inf]),
                high=np.array([1.0, 1.0, np.inf, np.inf]),
                dtype=np.float32,
            ) for i in range(len(self.kinematic_rects))
        }
        obs_space_dict.update(obs_platforms_dict)
        obs_space = spaces.Dict(obs_space_dict)

        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "desired_goal": desired_goal_space,
            "achieved_goal": achieved_goal_space,
        })

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.use_simpler_goal = use_simpler_goal

        self._interrupted = False

        self.reset()

    def _create_static_b2_rect(self, center_x, center_y, width, height):
        self.world.CreateStaticBody(
            position=(center_x, center_y),
            shapes=b2PolygonShape(box=(0.5 * width, 0.5 * height)),
        )
        self.static_rects.append((center_x, center_y, width, height))

    def _create_kinematic_b2_rect(self, pos_func, vel_func, width, height, color):
        body = self.world.CreateKinematicBody(position=(0.0, 0.0),
                                              shapes=b2PolygonShape(box=(0.5 * width, 0.5 * height)))
        self.kinematic_rects.append([body, pos_func, vel_func, width, height, color])

    def _draw_goal_return(self):
        random_x = np.random.uniform(-0.1, 0.1)
        if self.use_simpler_goal:
            mean_x = 0.75
            y = -0.1
        else:
            mean_x = -0.25
            if self.third_platform == [True]:
                mean_x = -0.75
                y = self.level_3 + self.goal_radius
            else:
                y = self.level_2 + self.goal_radius
        mean_x += random_x
        return np.array((mean_x, y))

    def _draw_goal(self):
        self.goal = self._draw_goal_return()

    def compute_reward(self, achieved_goal, desired_goal, info):
        if (np.linalg.norm(achieved_goal - desired_goal) <= self.goal_radius / self.level_width / 0.5):
            return 0.0
        else:
            return -1.0

    def _get_obs(self):
        agent_obs = {
            "position": np.array(self.ball.position) / self.level_width * 2.0,
            "velocity": np.array(self.ball.linearVelocity) / self.max_vel_comp,
            "ang_vel": np.array([self.ball.angularVelocity / self.max_ang_vel]),
        }
        platf_obs = {
            "platform{}".format(i): np.concatenate([
                platf[0].position / self.level_width * 2.0,
                platf[0].linearVelocity / self.max_vel_comp,
            ]) for i, platf in enumerate(self.kinematic_rects)
        }
        partial_obs = {**agent_obs, **platf_obs}
        obs = {
            "observation": partial_obs,
            "desired_goal": self.goal / self.level_width * 2.0,
            "achieved_goal": np.array(self.ball.position) / self.level_width * 2.0,
        }
        return obs

    def step(self, action):
        self.ball.ApplyForce((action[0] * self.max_force, 0.0), point=self.ball.position, wake=True)

        t = self.current_step * self.time_step
        for k_rect in self.kinematic_rects:
            k_rect[0].position = k_rect[1](t)
            k_rect[0].linearVelocity = k_rect[2](t)

        # run integrator and constraint solver in Box2D
        self.world.Step(self.time_step, self.vel_iters, self.pos_iters)
        self.world.ClearForces()

        # clip linear and angular velocity
        self.ball.linearVelocity = np.clip(
            self.ball.linearVelocity,
            -self.max_vel_comp + self._eps,
            self.max_vel_comp - self._eps,
        )
        self.ball.angularVelocity = np.clip(
            self.ball.angularVelocity,
            -self.max_ang_vel + self._eps,
            self.max_ang_vel - self._eps,
        )

        # update triggers
        for trigger in self.triggers:
            trigger.update(self.ball.position, 1.0)

        if self.triggers[0].is_frozen:
            self.kinematic_rects[0][5] = self.frozen_trigger_color
        else:
            self.kinematic_rects[0][5] = self.platform_color

        if self.triggers[1].is_frozen:
            self.kinematic_rects[1][5] = self.frozen_trigger_color
        elif self.triggers[1].is_active:
            self.kinematic_rects[1][5] = self.active_trigger_color
        else:
            self.kinematic_rects[1][5] = self.inactive_trigger_color

        self.current_step += 1
        info = {}
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        done = reward == 0.0 or self.current_step >= self.max_episode_length
        if done:
            info['TimeLimit.truncated'] = reward != 0.0
        return obs, reward, done, info

    def reset(self):
        self.ball.position = self.agent_initial_position
        self.ball.angle = 0.0
        self.ball.linearVelocity = np.zeros((2,))
        self.ball.angularVelocity = 0.0

        for trigger in self.triggers:
            trigger.reset()

        self.current_step = 0
        self._draw_goal()
        return self._get_obs()

    def update_subgoals(self, subgoals):
        self._subgoals = subgoals

    def update_timed_subgoals(self, timed_subgoals, tolerances, interrupted: bool = False):
        self._timed_subgoals = timed_subgoals
        self._tolerances = tolerances
        self._interrupted = interrupted

    def render(self, mode="human", close=False):
        import pyglet
        import pyglet.gl as gl

        from ..utils.pyglet_utils import (
            draw_circle_sector,
            draw_box,
            draw_line,
            draw_vector,
            draw_vector_with_outline,
            draw_circular_subgoal,
        )

        if self.window is None:
            self.window = pyglet.window.Window(
                width=self.window_width,
                height=self.window_height,
                vsync=True,
                resizable=True,
            )
            gl.glClearColor(*self.background_color)

        @self.window.event
        def on_resize(width, height):
            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(
                -0.5 * self.level_width,
                0.5 * self.level_width,
                -0.5 * float(height) / width * self.level_width,
                0.5 * float(height) / width * self.level_width,
                -1.0,
                1.0,
            )
            gl.glMatrixMode(gl.GL_MODELVIEW)
            return pyglet.event.EVENT_HANDLED

        def draw_timed_circular_subgoal(position, velocity, delta_t_ach, delta_t_comm, radius, color, arrow_config):
            # desired time until achievement
            draw_box(
                position + (0.0, radius + 0.04),
                delta_t_ach / 100.0 + 0.02,
                0.03 + 0.02,
                0.0,
                (0.0, 0.0, 0.0),
            )
            draw_box(position + (0.0, radius + 0.04), delta_t_ach / 100.0, 0.03, 0.0, color)
            # subgoal
            draw_circular_subgoal(position, velocity, radius, color, arrow_config)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        gl.glLoadIdentity()

        n_triangles = 32

        # static rectangles making up level
        for rect in self.static_rects:
            draw_box(rect[0:2], rect[2], rect[3], 0.0, self.static_color)

        # kinematic rectangles making up platforms
        for k_rect in self.kinematic_rects:
            draw_box(k_rect[0].position, k_rect[3], k_rect[4], 0.0, k_rect[5])

        # trigger
        tr = self.triggers[1]
        draw_box(
            (tr.pos[0], tr.pos[1] - tr.height),
            tr.width,
            tr.height,
            0.0,
            self.active_trigger_color if tr.is_active else self.inactive_trigger_color,
        )

        # goal
        draw_circle_sector(self.goal, 0.0, self.goal_radius, n_triangles, self.goal_color, n_triangles)
        draw_circle_sector(
            self.goal,
            0.0,
            self.goal_radius * 0.8,
            n_triangles,
            self.background_color[:3],
            n_triangles,
        )

        # subgoals
        for sg, color, ac in zip(self._subgoals, self._subgoal_colors, self._subgoal_acs):
            if sg is not None:
                draw_circular_subgoal(
                    sg["position"],
                    sg["velocity"] * self.max_vel_comp,
                    self.subgoal_radius,
                    color,
                    ac,
                )

        # timed subgoals
        for ts, color, ac, tol in zip(
                self._timed_subgoals,
                self._subgoal_colors,
                self._subgoal_acs,
                self._tolerances,
        ):
            if ts is not None:
                r = tol["position"] if tol is not None else self.subgoal_radius
                draw_timed_circular_subgoal(
                    ts.goal["position"],
                    ts.goal.get("velocity", np.array([0, 0])) * self.max_vel_comp,
                    ts.delta_t_ach,
                    ts.delta_t_comm,
                    r,
                    color if not self._interrupted else (0.9, 0.0, 0.1),
                    ac,
                )

        # agent
        draw_circle_sector(
            self.ball.position,
            self.ball.angle,
            self.agent_radius,
            n_triangles,
            self.agent_color,
            n_triangles,
        )
        draw_line(self.ball.position, self.agent_radius, self.ball.angle, self.line_color)
        draw_vector(self.ball.position, self.ball.linearVelocity, self._agent_ac)

        self.window.flip()

        if mode == "rgb_array":
            pyglet.image.get_buffer_manager().get_color_buffer().save("ds.png")
            image = plt.imread("ds.png")
            os.remove("ds.png")
            return (image * 255).astype(np.uint8)[:, :, [2, 1, 0, 3]]

    @staticmethod
    def create_generic_wandb_dict(
        action_init_hl_action,
        action_timestep_now,
    ):
        return {
            "hl_action_goal/ongoing_position_x":
                action_init_hl_action["goal"]["position"][0],
            "hl_action_goal/ongoing_position_y":
                action_init_hl_action["goal"]["position"][1],
            "hl_action_goal/new_position_x":
                action_timestep_now["goal"]["position"][0],
            "hl_action_goal/new_position_y":
                action_timestep_now["goal"]["position"][1],
            "hl_action_goal/distance_between_goal_pos":
                np.linalg.norm(action_init_hl_action["goal"]["position"] - action_timestep_now["goal"]["position"])
        }
