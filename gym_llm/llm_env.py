from abc import abstractmethod


class LLMEnv:

    @abstractmethod
    def get_observation_schema(self):
        ...

    @abstractmethod
    def get_goal_description(self):
        ...

    @abstractmethod
    def get_action_schema(self):
        ...