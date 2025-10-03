from gym.envs.registration import register

register(
    id='mrnav-v1',
    entry_point='gym_env.envs:mrnav',
    max_episode_steps=1000,  # Default max episode steps
)
