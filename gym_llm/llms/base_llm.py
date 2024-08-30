from abc import abstractmethod, ABC
from collections import deque


class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float = 0.8, system_prompt: str = '', history_len: int = 10):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.history = deque(maxlen=history_len)

    @abstractmethod
    def generate(self):
        ...

    def reset(self):
        self.history.clear()