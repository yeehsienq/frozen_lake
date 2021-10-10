from gym.envs.registration import register

register(
    id='frozen-lake-v0',
    entry_point='frozen_lake.envs:FrozenLakeEnv',
)
register(
    id='frozen-lake-large-v0',
    entry_point='frozen_lake.envs:FrozenLakeLargeEnv',
)