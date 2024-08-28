import yaml
from pathlib import Path
from typing import Dict
import gymnasium as gym

import gym_llm


def parse_config(path: Path) -> Dict:
    if not path.exists():
        raise ValueError(f'Config file {path} not found')

    return yaml.safe_load(open(path, 'r'))

def get_env(env_config: Dict):
    name = env_config.get('name', '')
    kwargs = env_config.get('kwargs', {})

    if kwargs is None:
        kwargs = {}

    if 'render_mode' not in kwargs:
        kwargs['render_mode'] = 'human'

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