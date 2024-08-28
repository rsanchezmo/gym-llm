from distutils.command.config import config

# Gym LLM
The purpose of this repository is to facilitate the deployment of LLM-powered agents that interacts with `gymnasium` environments by reflecting on its previous observations and actions.
The agents can be powered by `OpenAI` models or any local model that is integrated in `Ollama`. 

![gym-llm](./doc/gym-llm.png)


> [!NOTE] 
> As it will depend on your hardware, the model size and the fact that LLMs are not real time (high action rates: e.g. 20fps) this repo does not care about it. This type of agents will be suitable for high level control and planning, with low action rates!

The library offers:
- `LLMWrapper`: A wrapper for `gymnasium` environments so you define the minimum requirements for the agent to interact with the environment: observation schema, action schema and goal description.
- `Agent`: An agent class to ask for actions based on the previous observations and actions.

## Installation
To install the `gym-llm` package, run the following command:
```bash
cd gym-llm
pip install .
```

## Environments
Every environment must inherit from both `gymnasium.Env` and `gym_llm.LLMWrapper` to implement every `abstractmethod`.

- You must define `get_observation_schema()`, `get_action_schema()` and `get_goal_description()` so the agent
can receive that information as the `system_prompt`.
- To make it easy for the agent, it can only receive `gymnasium.spaces.Dict` observations. In that way, the `observation_schema` comes to help with further description of each field. The decision to not accept 
pure `gymnasium.spaces.Box` type is to guarantee a richer and more understandable observation prompt. 
- The actions can only be `gymnasium.spaces.Discrete` for now, as LLM fail on float numbers (e.g. 9.11 is higher than 9.9?). It may be extended in the future to other spaces.

## Configuration file
```yaml
agent:
    backend: 'ollama'             # 'ollama' or 'openai'
    model: 'llama3.1'             # llm model, should be available on the backend
    temperature: .8               # creativity
    history: 10                   # history window
    action_rate: 1                # how many frames to skip without any action

environment:
    name: 'LunarLanderLLM-v2'     # name of the registered environment
    kwargs: null                  # extra args for the environment
  
experiment:
    parent: 'experiments'         # parent folder
    name: 'lunar_lander_llama3.1' # experiment name
    use_datetime: true            # append datetime to the name
    seed: 0                       # seed for reproducibility
    save_gif: true                # save gif of the run
    gif_fps: 30                   # gif frames per second
    num_runs: 1                   # number of runs
    verbose: true                 # print information  
```

## Usage
You can create your custom loop to interact with the environment and the agent. Here is an example:
```python
import your_env_registry
import gym_llm

config = gym_llm.parse_config(path=Path('./configs/lunar_lander.yaml'))
env = gym_llm.get_env(env_config=config.get('environment'), render_mode='human')
agent = gym_llm.Agent(config=config.get('agent'),
              **gym_llm.get_env_definition(env))

obs, _ = env.reset(seed=seed)
agent.reset(seed=seed)
done = False
total_reward = 0
num_steps = 0

while not done:
    raw_output, action = agent.get_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    num_steps += 1
    done = terminated or truncated
```

Or you can use the `run_experiment` function to run the experiment with the configuration file as shown in the `main.py` file. Just change the configuration file path to your own.

```bash
python main.py
```
If using this utility function, there will be created a folder with the gif of each run, the raw outputs of the llm, the results metrics and the configuration file used.

## Results
| **Environment** |    **LLM**    | **Reward** | **Action rate** | **Initial seed** |
|:---------------:|:-------------:|:----------:|:---------------:|:----------------:|
| LunarLander-v2  | `llama3.1-8B` |    0.0     |        1        |        0         |


## Future work
- OpenAI integration
- Better rendering: img and llm reasoning on the same window
- Comparison against reinforcement learning agents
- Run batch of experiments

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