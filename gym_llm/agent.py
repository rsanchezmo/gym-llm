import gymnasium as gym

from typing import Dict

from llms.ollama_llm import OllamaLLM
from llms.openai_llm import OpenAILLM


class Agent:
    def __init__(self, config: Dict):
        self.observation_schema = None
        self.goal_description = None

        self.llm = None
        backend = config.get('backend', 'ollama')
        if backend == 'ollama':
            self.llm = OllamaLLM()
        elif backend == 'openai':
            self.llm = OpenAILLM()
        else:
            raise ValueError('Unknown LLM backend')

    def get_action(self, observation: gym.spaces.Dict):
        ...


    def reset(self):
        ...