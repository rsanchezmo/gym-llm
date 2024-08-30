import gymnasium as gym


from .lunar_lander import LunarLander
gym.register(
    id='LunarLanderLLM-v2',
    entry_point='environments:LunarLander',
    max_episode_steps=80,
    reward_threshold=200,
)

from .blackjack import BlackjackEnv
gym.register(
    id='BlackjackLLM-v1',
    entry_point='environments:BlackjackEnv',
    kwargs={"sab": True, "natural": False}
)

from .taxi import TaxiEnv
gym.register(
    id="TaxiLLM-v3",
    entry_point="environments:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=50,
)