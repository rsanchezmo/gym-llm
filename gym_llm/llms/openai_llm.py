from openai import OpenAI
import json
from gym_llm.logger import get_logger
from gym_llm.llms.base_llm import BaseLLM



class OpenAILLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.8, system_prompt: str = '', history_len: int = 10):
        super().__init__(model=model, temperature=temperature, system_prompt=system_prompt, history_len=history_len)

        self.openai_client = OpenAI()

    def generate(self):
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'system', 'content': self.system_prompt}] + list(self.history),
                temperature=self.temperature,
                max_tokens=256,
                response_format={"type": "json_object"}
            )

            message_content = response.choices[0].message.content

            try:
                answer_json = json.loads(message_content)
                action = answer_json.get('action', None)
                reflection = answer_json.get('reflection', '')
            except json.JSONDecodeError:
                action = None
                reflection = ''
                get_logger().warn(f'Error in parsing the message content as JSON: {message_content}')

        except Exception as e:
            action = None
            reflection = ''
            get_logger().warn(f'Error in generation: {e}')

        content = {
            'reflection': reflection,
            'action': action
        }

        return content