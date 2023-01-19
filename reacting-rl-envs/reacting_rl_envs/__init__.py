import gym
from .hit_target import HitTargetEnv


gym.register(id='HitTargetEnv-v0', entry_point=HitTargetEnv, max_episode_steps=1000)
