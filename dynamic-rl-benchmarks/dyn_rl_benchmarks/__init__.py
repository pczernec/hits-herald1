from gym.envs.registration import register


from .envs.drawbridge import DrawbridgeEnv
from .envs.tennis_2d import Tennis2DEnv


register(
    id="Drawbridge-v1",
    entry_point="dyn_rl_benchmarks.envs:DrawbridgeEnv",
    # max_episode_steps=DrawbridgeEnv.max_episode_length,  the limit is already handled by the environment
)

register(
    id="NoisyDrawbridge-v1",
    entry_point="dyn_rl_benchmarks.envs:DrawbridgeEnv",
    kwargs={"starts_noise_list": [200, 600], "use_boolean_phase": True},
)

register(
    id="BooleanDrawbridge-v1",
    entry_point="dyn_rl_benchmarks.envs:DrawbridgeEnv",
    kwargs={"use_boolean_phase": True},
)

register(
    id="CrashDrawbridge-v1",
    entry_point="dyn_rl_benchmarks.envs:DrawbridgeEnv",
    kwargs={"penalize_crash": True},
)


register(id="Tennis2D-v1", entry_point="dyn_rl_benchmarks.envs:Tennis2DEnv")

register(
    id="Tennis2DDenseReward-v1",
    entry_point="dyn_rl_benchmarks.envs:Tennis2DDenseRewardEnv",
)

register(id="Platforms-v1", entry_point="dyn_rl_benchmarks.envs:PlatformsEnv")

register(id="ThreePlatforms-v1", entry_point="dyn_rl_benchmarks.envs:PlatformsEnv")

register(
    id="NoisyPlatforms-v1",
    entry_point="dyn_rl_benchmarks.envs:PlatformsEnv",
    kwargs={"use_freeze": (True, True)},
)

register(id="EasyPlatforms-v1", entry_point="dyn_rl_benchmarks.envs:PlatformsEnv", kwargs={"use_simpler_goal": True})

register(id="PlatformsTime-v1", entry_point="dyn_rl_benchmarks.envs:PlatformsTimeEnv")
