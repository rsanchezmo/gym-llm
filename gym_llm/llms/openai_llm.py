from .base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.8):
        super().__init__(model=model, temperature=temperature)

    def generate(self):
        pass