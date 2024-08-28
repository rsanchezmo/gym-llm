from tabnanny import verbose

import yaml
from pathlib import Path
from typing import Dict
import gymnasium as gym
import imageio
from datetime import datetime
import gym_llm
import json


def parse_config(path: Path) -> Dict:
    if not path.exists():
        raise ValueError(f'Config file {path} not found')

    return yaml.safe_load(open(path, 'r'))

def get_env(env_config: Dict, render_mode: str = 'human'):
    name = env_config.get('name', '')
    kwargs = env_config.get('kwargs', {})

    if kwargs is None:
        kwargs = {}

    if 'render_mode' not in kwargs:
        kwargs['render_mode'] = render_mode

    return gym.make(id=name, **kwargs)

def get_env_definition(env):
    if not isinstance(env.unwrapped, gym_llm.LLMWrapper):
        raise ValueError(f'Expected an instance of LLMWrapper, got {type(env)}')

    if not isinstance(env, gym.Env):
        raise ValueError(f'Expected an instance of gym.Env, got {type(env)}')

    return {'observation_schema': env.unwrapped.get_observation_schema(),
            'action_schema': env.unwrapped.get_action_schema(),
            'goal_description': env.unwrapped.get_goal_description()
            }


def run_experiment(config):

    exp_config = config.get('experiment')

    exp_save_path = Path(exp_config.get('parent')) / exp_config.get('name')

    if exp_config.get('use_datetime', False):
        exp_save_path = exp_save_path / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    exp_save_path.mkdir(parents=True, exist_ok=True)

    # save the config file
    with open(exp_save_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    seed = exp_config.get('seed', 0)
    num_runs = exp_config.get('num_runs', 1)
    save_gif = exp_config.get('save_gif', False)
    gif_fps = exp_config.get('gif_fps', 30)
    verbose = exp_config.get('verbose', False)

    render_mode = 'human'
    if save_gif:
        render_mode = 'rgb_array'

    env = get_env(env_config=config.get('environment'), render_mode=render_mode)

    agent = gym_llm.Agent(config=config.get('agent'),
                  **get_env_definition(env))

    total_rewards = []
    total_steps = []


    for i in range(num_runs):
        obs, _ = env.reset(seed=seed)
        agent.reset(seed=seed)

        seed += 1

        done = False
        total_reward = 0
        num_steps = 0

        imgs = []
        raw_outputs = {}

        while not done:
            raw_output, action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            if save_gif:
                imgs.append(env.render())

            total_reward += reward
            num_steps += 1
            done = terminated or truncated

            if verbose:
                print('********************************')
                print(f'    Observation: {obs}')
                print(f'    Reflection: {raw_output["reflection"]}')
                print(f'    Action: {raw_output["action"]} -> {agent.action_schema.get(raw_output["action"], "Unknown action")}')
                print(f'    Reward: {reward} -> Total reward: {total_reward}')
                print(f'    Num steps: {num_steps}')
                print(f'    Terminated: {terminated}')
                print(f'    Truncated: {truncated}')

            raw_outputs[num_steps] = raw_output

        if verbose:
            print('********************************')
            print(f'    Total reward: {total_reward}')
            print(f'    Num steps: {num_steps}')

        if save_gif:
            imageio.mimsave(exp_save_path / f'run_{i}_seed_{seed-1}.gif',
                            imgs, fps=gif_fps)

        with open(exp_save_path / f'run_{i}_seed_{seed-1}_raw_outputs.json', 'w') as f:
            json.dump(raw_outputs, f, indent=4)

        total_rewards.append(total_reward)
        total_steps.append(num_steps)

    avg_reward = sum(total_rewards) / num_runs
    avg_steps = sum(total_steps) / num_runs

    # save results in a json file
    with open(exp_save_path / 'results.json', 'w') as f:
        json.dump({
            'total_rewards': total_rewards,
            'total_steps': total_steps,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        }, f, indent=4)

    if verbose:
        print('********************************')
        print(f'    Average reward: {avg_reward}')
        print(f'    Average steps: {avg_steps}')