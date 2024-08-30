from .base_llm import BaseLLM
import ollama
import json
from gym_llm.logger import get_logger


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.8, system_prompt: str = '', history_len: int = 10):

        super().__init__(model=model, temperature=temperature, system_prompt=system_prompt, history_len=history_len)

        self.ollama_options = ollama.Options(
            temperature=temperature,
            num_predict=256,
            num_ctx=4096,
        )

    def generate(self):
        answer = ollama.chat(
            model=self.model,
            options=self.ollama_options,
            messages=[{'role': 'system', 'content': self.system_prompt}] + list(self.history),
            format='json'
        )


        try:
            answer_json = json.loads(answer['message']['content'])
            action = answer_json.get('action', None)
            reflection = answer_json.get('reflection', '')
        except Exception as e:
            action = None
            reflection = ''
            get_logger().warn(f'Error in generation: {e}')


        content = {
            'reflection': reflection,
            'action': action
        }

        return content


