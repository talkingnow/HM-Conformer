import os

from ._interface import Logger
from .local_logger import LocalLogger
from .neptune import NeptuneLogger
from .wandb import WandbLogger

class LoggerList(Logger):
    def __init__(self, loggers):
        self.loggers = loggers
    
    def log_metric(self, name, value, step=None):
        for logger in self.loggers:
            logger.log_metric(name, value, step)

    def log_text(self, name, text):
        for logger in self.loggers:
            logger.log_text(name, text)

    def log_arguments(self, dictionary):
        for logger in self.loggers:
            logger.log_arguments(dictionary)

    def log_image(self, name, image):
        for logger in self.loggers:
            logger.log_image(name, image)

    def save_model(self, name, model):
        for logger in self.loggers:
            logger.save_model(name, model)

    class Builder():
        def __init__(self, name, project, tags=None, description=None, scripts=None, args=None):
            self.loggers = []
            self.name = name
            self.project = project
            self.tags = tags
            self.args = args
            self.description = description
            self.scripts = scripts
        
        def use_local_logger(self, path):
            path = os.path.join(path, self.project, self.name)
            self.loggers.append(LocalLogger(path, self.description, self.scripts))
        
        def use_neptune_logger(self, user, token):
            self.loggers.append(
                NeptuneLogger(user, token, self.name, self.project, self.tags, self.description, self.scripts)
            )
        def use_wandb_logger(self, entity, api_key, group):
            self.loggers.append(
                WandbLogger(entity, api_key, group, self.name, self.project, self.tags, self.description, self.scripts, self.args)
            )
        
        def build(self):
            return LoggerList(self.loggers)