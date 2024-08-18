import gymnasium as gym

from typing import Dict

from gym_llm.llms.ollama_llm import OllamaLLM
from gym_llm.llms.openai_llm import OpenAILLM


class Agent:
    def __init__(self,
                 config: Dict,
                 observation_schema: str,
                 goal_description: str,
                 action_schema: str):

        self.llm = None

        self.system_prompt = (f'You are an agent that interacts with an environment.\n'
                              f'You must act following this task: {goal_description}.\n'
                              f'You will receive your observation and action history. You must first reflect and then generate the next action.\n'
                              f'The observations will come following this observation schema: {observation_schema}.\n'
                              f'The actions must be generated following this action schema: {action_schema}. You must answer with the key of that schema, not with the meaning of the key\n'
                              f'You must generate a JSON object with the following content:'
                              '\n{'
                              '\n  "reflection": <reflection>,'
                              '\n  "action": <action>'
                              '\n}\n'
                              'The reflection must be one line length and must be a sequence of thoughts that leads you to the final action!!')

        backend = config.get('backend', 'ollama')
        model = config.get('model', 'llama3.1')
        temperature = config.get('temperature', 0.2)
        self.action_rate = config.get('action_rate', 1)
        self.action_count = -1
        self.last_action = None

        if backend == 'ollama':
            self.llm = OllamaLLM(model=model, temperature=temperature)
        elif backend == 'openai':
            self.llm = OpenAILLM(model=model, temperature=temperature)
        else:
            raise ValueError('Unknown LLM backend')

    def get_action(self, observation: gym.spaces.Dict):
        self.action_count += 1

        self.llm.history.append(
            {
                'role': 'user',
                'content':
                    str({
                        'observation': str(observation),
                        'start': self.action_count == 0
                    })
            }
        )

        if self.action_count % self.action_rate == 0:
            result = self.llm.generate()

            # TODO: add validation of the result, both reflection and action must be present and the action must be a spaces discrete
            return result, int(result['action'])

        # TODO: last action!!!
        return {'reflection': 'Using last action', 'action': self.last_action}, self.last_action

    def reset(self, seed: int | None):
        self.action_count = -1
        self.llm.reset(self.system_prompt)
        self.last_action = None
