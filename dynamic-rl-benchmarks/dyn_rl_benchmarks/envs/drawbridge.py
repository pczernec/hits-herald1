import os
import random
from typing import Optional, List

import numpy as np

import gym
from gym import spaces


class DrawbridgeEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    max_episode_length = 1000
    _goal_radius = 1.0
    _river_length = 10.0
    _goal = _river_length - _goal_radius
    _drawbridge_start = 500.0
    _unfurl_speed = 0.03
    _sail_drag = 0.0001
    _max_vel = 0.03

    def __init__(self, subgoal_radius = 0.05,
                 starts_noise_list:Optional[List[int]]=None,
                 use_boolean_phase:bool=False,
                 penalize_crash:bool=False):
        super().__init__()

        self.penalize_crash = penalize_crash
        self.starts_noise_list = starts_noise_list
        self.bridge_start_noise = None
        self.use_boolean_phase = use_boolean_phase

        desired_goal_space = spaces.Box(
                low = -1.,
                high = 1.,
                shape = (1,),
                dtype = np.float32)
        achieved_goal_space = desired_goal_space

        box_space_1d = spaces.Box(
                low = -1.,
                high = 1.,
                shape = (1,),
                dtype = np.float32)

        obs_space = spaces.Dict({
            "ship_pos": box_space_1d,
            "ship_vel": box_space_1d,
            "sails_unfurled": box_space_1d,
            "bridge_phase": box_space_1d
            })
        if self.use_boolean_phase:
            obs_space = spaces.Dict({
                "ship_pos": box_space_1d,
                "ship_vel": box_space_1d,
                "sails_unfurled": box_space_1d,
                "bridge_phase": box_space_1d,
                "bridge_phase1": box_space_1d
            })

        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "desired_goal": desired_goal_space,
            "achieved_goal": achieved_goal_space
            })

        self.action_space = spaces.Box(
                low = -1.,
                high = 1.,
                shape = (1,),
                dtype = np.float32)

        self.window = None

        self.window_width = 1920
        self.window_height = 1080
        self.background_color = (1.0, 1.0, 1.0, 1.0)
        self._position = (5.0, 3.0, -13.0)
        self._lookat = (0., 0., -7.3, 0.)

        self.current_step = 0

        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = []
        self.subgoal_radius = float(subgoal_radius)

        self._env_geoms = ["riverbank", "water", "bridge_base", "underground"]
        self._n_sails = 8


    def compute_reward(self, achieved_goal, desired_goal, info):
        if abs(achieved_goal[0] - desired_goal[0]) <= \
                self._goal_radius/self._river_length:
                    return 0.
        else:
            return -1.

    def _get_obs(self):
        bridge_phase = [2.*self.current_step/self.max_episode_length - 1.]
        partial_obs = {
                "ship_pos": [2.*self.ship_pos/self._river_length - 1.],
                "ship_vel": [self.ship_vel/self._max_vel],
                "sails_unfurled": [self.sails_unfurled*2. - 1.],
                "bridge_phase": bridge_phase
                }
        if self.use_boolean_phase:
            bridge_phase = [int(self._is_drawbridge_open())]
            bridge_phase1 = [int(self._is_drawbridge_opening())]
            partial_obs["bridge_phase"] = bridge_phase
            partial_obs["bridge_phase1"] = bridge_phase1
        obs = {
            "observation": partial_obs,
            "desired_goal": [2.*self._goal/self._river_length - 1.],
            "achieved_goal": partial_obs["ship_pos"],
            }
        return obs

    def step(self, action):
        self.sails_unfurled += action[0]*self._unfurl_speed
        self.sails_unfurled = np.clip(self.sails_unfurled, 0.0, 1.0)
        self.ship_pos += self.ship_vel
        self.ship_pos = np.clip(self.ship_pos, 0., self._river_length)
        self.ship_vel += self._sail_drag*self.sails_unfurled
        self.ship_vel = np.clip(self.ship_vel, -self._max_vel, self._max_vel)

        crash_vel = 0
        if not self._is_drawbridge_open() and self.ship_pos > 4.3:
            if self.ship_vel > 0.:
                crash_vel = self.ship_vel
                self.ship_vel = -0.1*self.ship_vel

        obs = self._get_obs()
        info = {}

        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        reward = self._add_crash_penalty(crash_vel, reward)

        self.current_step += 1
        done = reward == 0. or self.current_step >= self.max_episode_length
        if done:
            info['TimeLimit.truncated'] = reward != 0.0
        return obs, reward, done, info

    def _add_crash_penalty(self, crash_vel, reward):
        reward += -crash_vel / self._max_vel * 50 if self.penalize_crash else 0
        reward += -2 if crash_vel > 0 and self.penalize_crash else 0
        return reward

    def _is_drawbridge_open(self, forced_step:Optional[int]= None):
        if self._get_drawbridge_angle(forced_step) > -25.0:
            return False
        else:
            return True

    def _is_drawbridge_opening(self, forced_step:Optional[int]= None):
        if self._get_drawbridge_angle(forced_step) < -0:
            return True
        else:
            return False

    def _get_drawbridge_angle(self, forced_step:Optional[int]= None):
        diff = forced_step - self._drawbridge_start if forced_step else self.current_step - self._drawbridge_start
        return -min(max(diff, 0.)*0.4, 90.0)

    def _get_random_element_timestep(self):
        # Because we start from current_step in every env from 1
        prev = self._is_drawbridge_opening(1)
        for i in range(2, self.max_episode_length):
            # Herald should react on the opening of the bridge
            current = self._is_drawbridge_opening(i)
            if prev != current:
                return i
            prev = current

    def reset(self):
        self.ship_pos = 0.0
        self.ship_vel = 0.0
        self.sails_unfurled = 0.0
        self.current_step = 0
        if self.starts_noise_list is not None:
            self._drawbridge_start = random.sample(list(range(self.starts_noise_list[0],
                                                              self.starts_noise_list[1])), k=1)[0]
        return self._get_obs()

    def update_subgoals(self, subgoals):
        self._subgoals = subgoals

    def update_timed_subgoals(self, timed_subgoals, tolerances):
        self._timed_subgoals = timed_subgoals
        self._tolerances = tolerances

    def render(self, mode='human', close=False):
        from ..utils.graphics_utils import get_default_subgoal_colors
        import pyglet
        import pyglet.gl as gl
        from wavefront_reader import read_wavefront

        from ..utils.pyglet_utils import render_wavefront_geom, draw_circle_sector3d

        if self.window is None:
            self._subgoal_colors = get_default_subgoal_colors()
            self.window = pyglet.window.Window(width = self.window_width,
                                               height = self.window_height,
                                               vsync = True,
                                               resizable = True)
            gl.glClearColor(*self.background_color)

            # load meshes
            self.geoms = read_wavefront(os.path.join(os.path.dirname(__file__), "assets", "drawbridge.obj"))

            # position of sails
            self._sails_y = [np.max(self.geoms[f"sail{i}"]["v"][:, 1]) for i in range(1, self._n_sails + 1)]

            # y position of subgoal
            self._subgoal_y = np.min(self.geoms["subgoal"]["v"][:, 1])

            # lighting
            gl.glEnable(gl.GL_LIGHTING)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat*4)(0.1,0.1,0.1,1))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat*4)(1,1,1,1))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat*4)(0,0,0,1))
            gl.glEnable(gl.GL_LIGHT0)
            gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)

            # recalculate normals (needed due to scaling of subgoals and sails)
            gl.glEnable(gl.GL_NORMALIZE)

            # depth test
            gl.glEnable(gl.GL_DEPTH_TEST)

            # semi-transparent objects
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


        @self.window.event
        def on_resize(width, height):
            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.gluPerspective(40, width / float(height), .1, 1000)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            return pyglet.event.EVENT_HANDLED

        self.window.clear()
        self.window.dispatch_events()

        gl.glLoadIdentity()

        # camera transformation
        gl.gluLookAt(*self._position, *self._lookat, 1., 0.)

        # environment
        for geom_name in self._env_geoms:
            render_wavefront_geom(self.geoms[geom_name])

        def draw_ship(pos, sails_unfurled, color=None, alpha=1.0):
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.0, pos)
            render_wavefront_geom(self.geoms["ship"],color=color,alpha=alpha)
            for i in range(self._n_sails):
                # sails
                gl.glPushMatrix()
                gl.glTranslatef(0.0, self._sails_y[i], 0.0)
                gl.glScalef(1.0, sails_unfurled, 1.0)
                gl.glTranslatef(0.0, -self._sails_y[i], 0.0)
                render_wavefront_geom(self.geoms[f"sail{i + 1}"],color=color,alpha=alpha)
                gl.glPopMatrix()
            gl.glPopMatrix()

        # ship
        draw_ship(self.ship_pos, self.sails_unfurled)

        angle = self._get_drawbridge_angle()
        center_of_rot = (0.6, 0.275, 0.0)

        # drawbridge left
        gl.glPushMatrix()
        gl.glTranslatef(center_of_rot[0], center_of_rot[1], center_of_rot[2])
        gl.glRotatef(angle, 0.0, 0.0, 1.0)
        gl.glTranslatef(-center_of_rot[0], -center_of_rot[1], -center_of_rot[2])
        render_wavefront_geom(self.geoms["bridge_left"])
        gl.glPopMatrix()

        # drawbridge right
        gl.glPushMatrix()
        gl.glTranslatef(-center_of_rot[0], center_of_rot[1], center_of_rot[2])
        gl.glRotatef(-angle, 0.0, 0.0, 1.0)
        gl.glTranslatef(center_of_rot[0], -center_of_rot[1], -center_of_rot[2])
        render_wavefront_geom(self.geoms["bridge_right"])
        gl.glPopMatrix()

        def render_subgoal(sg, tol, color):
            # cull polygons facing away from camera
            gl.glEnable(gl.GL_CULL_FACE)
            pos = (sg["ship_pos"] + 1.0)*0.5*self._river_length
            sails_unfurled = (sg["sails_unfurled"] + 1.0)*0.5
            draw_ship(pos, sails_unfurled, color=color, alpha=0.4)
            gl.glDisable(gl.GL_CULL_FACE)

        # timed subgoals
        for ts, color, tol in zip(self._timed_subgoals, self._subgoal_colors,
                self._tolerances):
            if ts is not None:
                # translucent ship in goal state
                pos_tol = tol["ship_pos"] if tol is not None else self.subgoal_radius
                render_subgoal(ts.goal, pos_tol, color)
                # circle sector indicating desired time until achievement
                draw_circle_sector3d(
                    center=(0., 1.5, (ts.goal["ship_pos"] + 1.0)*0.5*self._river_length - 9.),
                    rotation=(90., 0., 0.),
                    angle=ts.delta_t_ach*0.02,
                    radius=0.3,
                    n=64,
                    color=color
                )

        # subgoals
        for sg, color in zip(self._subgoals, self._subgoal_colors):
            if sg is not None:
                render_subgoal(sg, self.subgoal_radius, color)

        self.window.flip()

