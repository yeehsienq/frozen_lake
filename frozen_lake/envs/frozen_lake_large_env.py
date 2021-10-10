import gym
from gym import error, spaces, utils
from gym.utils import seeding
from frozen_lake.envs.frozen_lake_env import FrozenLakeEnv

class FrozenLakeLargeEnv(FrozenLakeEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self, kwargs):
    super().__init__(kwargs, n=10)