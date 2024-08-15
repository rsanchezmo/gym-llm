# Gym LLM
The purpose of this repository is to facilitate the deployment of LLM-powered agents that interacts with `gymnasium` environments by reflecting on its previous observations and actions.
The agents can be powered by `OpenAI` models or any local model that is integrated in `Ollama`. 

The library offers:
- `LLMWrapper`: A wrapper for `gymnasium` environments so you define the minimum requirements for the agent to interact with the environment: observation schema and goal description.
- `Agent`: An agent class to ask for actions based on the previous observations and actions.

## Installation
To install the `gym-llm` package, run the following command:
```bash
cd gym-llm
pip install .
```

## Usage

## Results

| **Environment** |    **LLM**    | **Reward** |
|:---------------:|:-------------:|:----------:|
| LunarLander-v2  | `llama3.1-8B` |    0.0     |

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