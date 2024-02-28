from gymnasium.envs.registration import register
register(
    id='AddictiveEnv_v1',
    entry_point='AddictiveReward.envs:AddictiveEnv_v1',
)
register(
    id='AddictiveEnv_v2',
    entry_point='AddictiveReward.envs:AddictiveEnv_v2',
)
register(
    id='AddictiveEnv_v3',
    entry_point='AddictiveReward.envs:AddictiveEnv_v3',
)
