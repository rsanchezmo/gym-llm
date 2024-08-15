import gymnasium as gym
from gym_llm import Agent


if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    agent = Agent()

    obs, _ = env.reset()
    agent.reset()

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        img = env.render()
        done = terminated or truncated