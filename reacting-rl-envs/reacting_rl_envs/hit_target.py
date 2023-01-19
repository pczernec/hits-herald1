from time import sleep
from typing import Any, Mapping, Optional, Tuple, Union
import numpy as np
from gym import GoalEnv
from gym.spaces import Box, Dict
from Box2D import b2World, b2CircleShape, b2FixtureDef


def get_angle(vec):
    return np.arctan2(vec[1], vec[0])


class HitTargetEnv(GoalEnv):
    """
    An environment where the objective is to hit a target with the controlled object. 
    The environemnt is filled with obstacles that have to be avoided or will slow the object down considerably.

    The environment is partially observable - the controlled object does not know the layout of the obstacles.

    The action specifies angular velocity of the controlled object. The actual speed is not controllable

    The observation contains:
    object location x
    object location y
    object direction in range [-pi, pi)
    target location x
    target location y
    closeness (1/(1+dist)) in range [0-1] of the nearest obstacle in each view:
        0 - no obstacle in range for the given view
        1 - the obstacle is next to the controlled object
    """

    ObsType = Union[np.ndarray, dict]

    def __init__(
            self, view_range=0.2, num_views=8,
            object_size=0.025, num_obstacles=20, obstacle_size=0.05,
            speed=0.005, max_angular_velocity=0.2, min_obstacle_dist=0.1,
            target_dist = 0.9, default_target_loc=None,
            flat_obs=False) -> None:
        self.num_views = num_views
        self.view_range = view_range
        self.object_size=object_size
        self.num_obstacles=num_obstacles
        self.obstacle_size=obstacle_size
        self.speed = speed
        self.max_angular_velocity = max_angular_velocity
        self.min_obstacle_dist = min_obstacle_dist
        self.target_dist = target_dist
        self.flat_obs = flat_obs

        self.action_space = Box(-1, 1, shape=(1,))
        low_obs = np.array([-np.inf, -np.inf, -np.pi] + [0] * num_views)
        low_goal = np.array([-np.inf, -np.inf])
        high_obs = np.array([np.inf, np.inf, np.pi] + [1] * num_views)
        high_goal = np.array([np.inf, np.inf])
        if flat_obs:
            self.observation_space = Box(
                np.concatenate([low_obs, low_goal]),
                np.concatenate([high_obs, high_goal])
            )
        else:
            self.observation_space = Dict({
                'observation': Box(low_obs, high_obs),
                'desired_goal': Box(low_goal, high_goal),
                'achieved_goal': Box(low_goal, high_goal),
            })


        self.world = b2World(gravity=(0, 0))

        self.object = self.world.CreateDynamicBody(
            position=(0, 0),
            fixtures=b2FixtureDef(
                shape=b2CircleShape(pos=(0, 0), radius=object_size),
                restitution=0,
                density=1000,
                friction=0,
            )
        )
        self.target = np.array([1, 0])
        self.default_target_loc = default_target_loc
        self.obstacles = [self.world.CreateStaticBody(
            position=(1, 1),
            fixtures=b2FixtureDef(shape=b2CircleShape(pos=(0, 0), radius=obstacle_size))
        ) for _ in range(num_obstacles)]

        self.window = None

    def step(self, action: float) -> Tuple[ObsType, float, bool, bool, dict]:
        angle_diff = float(action * self.max_angular_velocity)
        angle = get_angle(self.object.linearVelocity)
        self.object.linearVelocity = np.cos(angle + angle_diff) * self.speed, np.sin(angle + angle_diff) * self.speed

        self.world.Step(1, 1, 0)

        reward = self.compute_reward(self.object.position, self.target, {})
        done = reward == 0

        info = {
            'object_position': self.object.position,
            'object_velocity': self.object.linearVelocity,
            'target_position': self.target,
            'obstacles': [obstacle.position for obstacle in self.obstacles]
        }

        return self._get_obs(), reward, done, info

    def reset(self) -> Tuple[ObsType, dict]:
        self.object.position = (0, 0)
        direction = np.random.uniform(-np.pi, np.pi)
        self.object.linearVelocity = (np.cos(direction) * self.speed, np.sin(direction) * self.speed)

        if self.default_target_loc:
            self.target = self.default_target_loc
            target_dist = np.linalg.norm(self.target)
        else:
            target_dist = self.target_dist
            target_angle = np.random.uniform(-np.pi, np.pi)
            self.target = np.array([np.cos(target_angle), np.sin(target_angle)]) * target_dist

        for obstacle in self.obstacles:
            obstacle_position = self.target
            while np.linalg.norm(obstacle_position - self.target) < max(self.object_size, self.obstacle_size):
                angle = np.random.uniform(-np.pi, np.pi)
                distance = np.random.uniform(self.min_obstacle_dist, target_dist)

                obstacle_position = np.array([np.cos(angle) * distance, np.sin(angle) * distance])
            
            obstacle.position = obstacle_position

        return self._get_obs()

    def _get_view(self, i):
        angle_from = 2 * np.pi / self.num_views * i
        angle_to = 2 * np.pi / self.num_views * (i + 1)

        dists = []
        object_angle = get_angle(self.object.linearVelocity)
        rotation_matrix = np.array([[np.cos(object_angle), -np.sin(object_angle)], [np.sin(object_angle), np.cos(object_angle)]])
        for obstacle in self.obstacles:
            relative = np.subtract(obstacle.position, self.object.position)
            # rotate relative center
            relative = np.matmul(rotation_matrix, relative)
            angle = get_angle(relative)

            if angle < 0:
                angle += 2 * np.pi
            if angle >= angle_from and angle < angle_to:
                # simple case - center of the obstacle is visible in the view
                dists.append(np.linalg.norm(relative) - self.obstacle_size)
            else:
                # check if the obstacle is visible, but not center.
                # that will be intersection of the line limiting the view and the circle
                # i.e., solution to 
                # x = d * cos a 
                # y = d * sin a
                # (x-cx) ^ 2 + (y-cy) ** 2 = r ^ 2
                # where x, y are intersection, a is the line angle, 
                # cx, cy are relative center of the obstacle,
                # r is the obstacle radius and d is the distance to the intersection.
                # 
                # the solution is the minimal positive solution to the equation:
                # d^2 - 2(cx * cos a + cy * sin a)*d + cx^2 + cy^2 - r^2 = 0

                for view_angle in angle_from, angle_to:
                    a = 1
                    b = 2 * (np.cos(view_angle) * relative[0] + np.sin(view_angle) * relative[1])
                    c = relative[0] ** 2 + relative[1] ** 2 - self.obstacle_size ** 2

                    delta = b ** 2 - 4 * a * c
                    if delta >= 0:
                        if (-b - np.sqrt(delta)) / (2 * a) > 0:
                            # smaller solution
                            dists.append((-b - np.sqrt(delta)) / (2 * a))
                        elif (-b + np.sqrt(delta)) / (2 * a) > 0:
                            dists.append((-b + np.sqrt(delta)) / (2 * a))

        if not dists or all(d > self.view_range for d in dists):
            return 0
        else:
            return 1 / (1 + min(dists) / self.view_range)

    def _get_obs(self):
        achieved = np.array(self.object.position)
        obs = np.array([
            *self.object.position,
            get_angle(self.object.linearVelocity),
            *[self._get_view(i) for i in range(self.num_views)]
        ])
        target = np.array(self.target)

        if self.flat_obs:
            return np.concatenate([obs, target])
        else:
            return {
                'observation': obs,
                'desired_goal': target,
                'achieved_goal': achieved,
            }

    def compute_reward(self, achieved_goal: object, desired_goal: object, info: Mapping[str, Any]) -> float:
        achieved = np.linalg.norm(achieved_goal - desired_goal) < self.unwrapped.object_size
        return 0 if achieved else -1

    def render(self, mode='human') -> Optional[np.ndarray]:
        import pyglet
        import pyglet.shapes as shapes

        window_size = 640

        if self.window is None:
            self.window = pyglet.window.Window(width=window_size, height=window_size, vsync=True)
            pyglet.gl.glClearColor(1, 1, 1, 1)

        self.window.dispatch_events()
        self.window.switch_to()
        self.window.clear()

        max_pos = max(np.abs(self.target) + self.object_size)
        for drawable in [self.object, *self.obstacles]:
            max_pos = max(*np.abs(drawable.position), max_pos)

        scale = window_size / (max_pos * 2)
        center = window_size / 2

        batch = pyglet.graphics.Batch()
        circles = []

        def draw_circle(position, size, color):
            circles.append(shapes.Circle(
                position[0] * scale + center,
                position[1] * scale + center,
                size * scale,
                color=color, batch=batch
            ))

        draw_circle(self.object.position, self.view_range, (255, 191, 191))
        draw_circle(self.object.position, self.object_size, (255, 32, 32))
        draw_circle(self.target, self.object_size, (0, 255, 0))
        for obstacle in self.obstacles:
            draw_circle(obstacle.position, self.obstacle_size, (64, 64, 64))

        line = shapes.Line(
            int(self.object.position[0] * scale + center),
            int(self.object.position[1] * scale + center), 
            int((self.object.position[0] + self.object.linearVelocity[0] * 0.15 / self.speed) * scale + center), 
            int((self.object.position[1] + self.object.linearVelocity[1] * 0.15 / self.speed) * scale + center), 
            width=2, color=(0, 0, 0, 255), batch=batch
        )
        line.opacity = 255

        batch.draw()
        if mode == "rgb_array":
            arr = np.frombuffer(
                pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().get_data("RGBA"), dtype=np.uint8
            ).reshape((window_size, window_size, 4))

        self.window.flip()

        if mode == "rgb_array":
            return arr

    def close(self):
        self.window.close()
        return super().close()


if __name__ == '__main__':
    env = HitTargetEnv()
    print(env.reset())
    env.render()
    for _ in range(500):
        act = env.action_space.sample()
        obs, rew, done, info = env.step(act)
        if done: 
            break
        print(obs)
        env.render()
        sleep(0.04)
    env.close()

