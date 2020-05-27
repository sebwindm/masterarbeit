from gym.envs.registration import register

register(
    id='jobshop-v0',
    entry_point='gym_jobshop.envs:JobShopEnv',
)