from pathlib import Path
import environments  # to register the environments

from gym_llm import parse_config, run_experiment

if __name__ == '__main__':

    config = parse_config(path=Path('./configs/taxi_llama31.yaml'))

    run_experiment(config=config)

