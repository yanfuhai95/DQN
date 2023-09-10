from abc import ABC, abstractmethod

class RLModel(ABC):
    @abstractmethod
    def optimize(self, state, action, reward, next_state):
        pass
    
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass
