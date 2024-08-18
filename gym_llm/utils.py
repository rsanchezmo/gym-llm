import yaml
from pathlib import Path
from typing import Dict
import gymnasium as gym



def parse_config(path: Path) -> Dict:
    if not path.exists():
        raise ValueError(f'Config file {path} not found')

    return yaml.safe_load(open(path, 'r'))

def get_env(env_config: Dict):
    name = env_config.get('name', '')
    max_episode_steps = env_config.get('max_episode_steps', 100)
    class_name = env_config.get('class_name', '')
    main_file = env_config.get('main_file', 'env.py')
    kwargs = env_config.get('kwargs', {})

    if kwargs is None:
        kwargs = {}

    if 'render_mode' not in kwargs:
        kwargs['render_mode'] = 'human'

    module_name = f"environments.{name}.{main_file.split('.py')[0]}"

    gym_id = f'gym_llm_{name}_env-v0'

    gym.register(id=gym_id,
             entry_point=f"{module_name}:{class_name}",
             max_episode_steps=max_episode_steps)


    return gym.make(id=gym_id, **kwargs)