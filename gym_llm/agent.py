import gymnasium as gym

from typing import Dict

from gym_llm.llms.ollama_llm import OllamaLLM
from gym_llm.llms.openai_llm import OpenAILLM
from gym_llm.logger import get_logger


class Agent:
    def __init__(self,
                 config: Dict,
                 observation_schema: str,
                 goal_description: str,
                 action_schema: str):

        self.llm = None

        self.system_prompt = f"""You are an intelligent agent interacting with a dynamic environment.
        Your task is to achieve the following goal: {goal_description}.
        You will be provided with the most recent observation and a flag indicating if it is the initial observation.
        Your first step is to analyze and reflect on each key in the observation to extract useful information for generating the next action.
        The observations will follow this structure: {observation_schema}.
        Actions must be generated based on this structure: {action_schema}.
        When deciding on an action, you should output the key corresponding to that action, not its description.
        You are required to produce a JSON object with the following format:
        {{
          "reflection": <short_reasoning>,
          "action": <action_key>
        }}
        Your reflection should be concise, serving as a brief chain of thought. Ensure that each reflection is unique and please, you must strictly consider the observation input for the reflection. 
        You have already been provided with the observation and action schema. Do not invent anything.
        
        Example answer for some use case:
        If the observation shows that the agent is near a wall and the goal is to move forward (0: left, 1: forward), the reflection might be:
        {{
        "reflection": "Since the agent is near a wall, moving forward might result in a collision. It would be better to turn left to avoid the obstacle and continue toward the goal."
        "action": "0"
        }}
        
        Another Example reasoning:
        If the observation shows that the agent is in an open area with the goal to reach a specific point ahead (0: left, 1: forward, 2: backward, 4: right), the reflection might be:
        {{
          "reflection": "The agent is in an open area with no obstacles in front, so moving forward is the most direct path to reach the goal. It does not make sense to go backward or turn left or right.",
          "action": "1"
        }}
        
        You must provide in the reflection key a reasoning of the effect of each of the actions (on the action schema) as depicted in the previous examples.
        You must try actions that avoids terminal states and try to reach the goal. As you are provided with history, you can use it to make better decisions by looking for cause-effect relationships.
        """

        backend = config.get('backend', 'ollama')
        model = config.get('model', 'llama3.1')
        temperature = config.get('temperature', 0.2)
        history_len = config.get('history', 10)

        self.action_rate = config.get('action_rate', 1)
        self.action_count = -1
        self.last_action = None

        if self.action_rate > history_len:
            raise ValueError('Action rate must be less than equal to history length')

        self.action_schema = action_schema

        if backend == 'ollama':
            self.llm = OllamaLLM(model=model, temperature=temperature, system_prompt=self.system_prompt, history_len=history_len)
        elif backend == 'openai':
            self.llm = OpenAILLM(model=model, temperature=temperature, system_prompt=self.system_prompt, history_len=history_len)
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

            if result['action'] is not None:
                try:
                    self.last_action = int(result['action'])
                except ValueError:
                    get_logger().warn(f"Action must be an integer, got {result['action']}")

                result['action'] = self.last_action
                content = result
            else:
                content = {
                    'reflection': '',
                    'action': self.last_action
                }

            self.llm.history.append(
                {
                    'role': 'assistant',
                    'content': str(content)
                }
            )

            return result, self.last_action

        return {'reflection': '', 'action': self.last_action}, self.last_action

    def reset(self, seed: int | None):
        self.action_count = -1
        self.llm.reset()
        self.last_action = None
