# Gym LLM
The purpose of this repository is to facilitate the deployment of LLM-powered agents that interacts with `gymnasium` environments by reflecting on its previous observations and actions.
The agents can be powered by `OpenAI` models or any local model that is integrated in `Ollama`. 

> [!NOTE] 
> As it will depend on your hardware, the model size and the fact that LLMs are not real time (high action rates: e.g. 20fps) this repo does not care about it. This type of agents will be suitable for high level control and planning, with low action rates!

The library offers:
- `LLMWrapper`: A wrapper for `gymnasium` environments so you define the minimum requirements for the agent to interact with the environment: observation schema, action schema and goal description.
- `Agent`: An agent class to ask for actions based on the previous observations and actions.

> [!WARNING] 
> This repository is yet under development and it is not ready for use. Hope to get it ready soon!

## Installation
To install the `gym-llm` package, run the following command:
```bash
cd gym-llm
pip install .
```

## Environments
Every environment must inherit from both `gymnasium.Env` and `gym_llm.LLMWrapper` to implement every abstractmethod.

- You must define `get_observation_schema()`, `get_action_schema()` and `get_goal_description()` so the agent
can receive that information as the `system_prompt`.
- To make it easy for the agent, it can only receive `gymnasium.spaces.Dict` observations. In that way, the `observation_schema` comes to help with further description of each field. The decision to not accept 
pure `gymnasium.spaces.Box` type is to guarantee a richer and more understandable observation prompt. 
- The actions can only be `gymnasium.spaces.Discrete` for now, as LLM fail on float numbers (e.g. 9.11 is higher than 9.9?). It may be extended in the future to other spaces.

## Configuration file
```yaml
agent:
    backend: 'ollama'           # 'ollama' or 'openai'
    model: 'llama3.1'           # llm model, should be available on the backend
    temperature: .8             # creativity
    history: 10                 # history window
    action_rate: 1              # how many frames to skip without any action

environment:
    name: 'lunar_lander'        # name of the environment folder
    max_episode_steps: 1000     # max episode steps
    class_name: 'LunarLander'   # name of the class of the env
    main_file: 'env.py'         # main file where the class is defined
    kwargs: null                # extra args for the environment

experiment:
    parent: 'experiments'
    name: 'lunar_lander_llama3.1'
    use_datetime: true
    seed: 0
```

## Usage

## Results

| **Environment** |    **LLM**    | **Reward** | **Action rate** |
|:---------------:|:-------------:|:----------:|:---------------:|
| LunarLander-v2  | `llama3.1-8B` |    0.0     |        1        |

## TODO
- Add skip frames, to avoid high correlation between frames

## Citation
If you find `gym-llm` useful, please consider citing:

```bibtex
  @misc{2024gymllm,
    title     = {Gym LLM},
    author    = {Rodrigo Sánchez Molina},
    year      = {2024},
    howpublished = {https://github.com/rsanchezmo/gym-llm},
  }
```