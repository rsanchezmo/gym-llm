from pathlib import Path
import gymnasium as gym
import environments  # to register the environments

from gym_llm import Agent, parse_config, get_env_definition, get_env

if __name__ == '__main__':

    config = parse_config(path=Path('./configs/lunar_lander.yaml'))

    env = get_env(env_config=config.get('environment'))

    agent = Agent(config=config.get('agent'),
                  **get_env_definition(env))

    exp_config = config.get('experiment')
    seed = exp_config.get('seed', 0)

    obs, _ = env.reset(seed=seed)
    agent.reset(seed=seed)

    done = False
    while not done:
        raw_output, action = agent.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        img = env.render()
        done = terminated or truncated