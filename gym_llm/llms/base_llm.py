from abc import abstractmethod, ABC


class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float = 0.8):
        self.model = model
        self.temperature = temperature
        self.history = []

    @abstractmethod
    def generate(self):
        ...

    def reset(self, system_prompt: str):
        self.history = [
            {
                'role': 'system',
                'content': system_prompt
            }
        ]