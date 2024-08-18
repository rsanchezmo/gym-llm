from abc import abstractmethod


class LLMWrapper:

    @abstractmethod
    def get_observation_schema(self):
        ...

    @abstractmethod
    def get_goal_description(self):
        ...

    @abstractmethod
    def get_action_schema(self):
        ...