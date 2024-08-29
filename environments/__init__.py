import gymnasium as gym


from .lunar_lander import LunarLander
gym.register(
    id='LunarLanderLLM-v2',
    entry_point='environments:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)