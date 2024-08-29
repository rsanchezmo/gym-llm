from .base_llm import BaseLLM
import ollama
import json
from gym_llm.logger import get_logger


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.8):

        super().__init__(model=model, temperature=temperature)

        self.ollama_options = ollama.Options(
            temperature=temperature,
            num_predict=256,
            num_ctx=4096,
        )

    def generate(self):
        answer = ollama.chat(
            model=self.model,
            options=self.ollama_options,
            messages=self.history,
            format='json'
        )


        try:
            answer_json = json.loads(answer['message']['content'])
            action = answer_json.get('action', None)
            reflection = answer_json.get('reflection', '')
        except Exception as e:
            action = None
            reflection = 'Error in generation of the json format'
            get_logger().warn(f'Error in generation: {e}')


        content = {
            'reflection': reflection,
            'action': action
        }

        self.history.append(
            {
                'role': 'assistant',
                'content': str(content)
            }
        )

        return content


