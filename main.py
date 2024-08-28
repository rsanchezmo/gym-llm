from pathlib import Path
from pprint import pprint
import environments  # to register the environments

from gym_llm import Agent, parse_config, get_env_definition, get_env, run_experiment

if __name__ == '__main__':

    config = parse_config(path=Path('./configs/lunar_lander.yaml'))

    run_experiment(config=config)

