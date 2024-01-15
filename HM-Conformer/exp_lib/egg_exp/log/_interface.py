from abc import ABCMeta, abstractmethod

class Logger(metaclass=ABCMeta):
    @abstractmethod
    def log_metric(self, name, value, step=None):
        pass

    @abstractmethod
    def log_text(self, name, text):
        pass

    @abstractmethod
    def log_arguments(self, dictionary):
        pass

    @abstractmethod    
    def log_image(self, name, image):
        pass

    @abstractmethod
    def save_model(self, name, state_dict):
        pass

    def finish(self):
        pass