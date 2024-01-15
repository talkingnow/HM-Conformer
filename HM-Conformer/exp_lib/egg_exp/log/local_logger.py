import os
import torch
import shutil

from ._interface import Logger

PATH = None

class LocalLogger(Logger):
    """Save experiment logs to local storage.
    """
    def __init__(self, path, description=None, scripts=None):
        # set path
        self.path = path
        
        # make directory
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        # save description
        if description is not None:
            self.log_text('description', description)

        # script backup
        if scripts is not None:
            shutil.copytree(scripts, f'{path}/scripts')
            
            for root, dirs, _ in os.walk(f'{path}/scripts'):
                for dir in dirs:
                    if dir == '__pycache__':
                        shutil.rmtree(f'{root}/{dir}')

    def log_text(self, name, text):
        f = f'{self.path}/{name}.txt'
        os.makedirs(os.path.dirname(f), exist_ok=True)
        mode = 'a' if os.path.exists(f) else 'w'
        f = open(f, mode, encoding='utf-8')
        f.write(text)
        f.close()
        
    def log_arguments(self, arguments):
        for k, v in arguments.items():
            self.log_text('parameters', f'{k}: {v}\n')

    def log_metric(self, name, value, step=None):
        if step is None:
            msg = f'{name}: {value}\n'
        else:
            msg = f'[{step}] {name}: {value}\n'

        self.log_text(name, msg)

    def log_image(self, name, image):
        pass
        # f = f'{self.path}/{name}.png'
        # os.makedirs(os.path.dirname(f), exist_ok=True)
        # image.save(f, 'PNG')
        
    def save_model(self, name, state_dict):
        # set model's results path
        path = os.path.join(self.path, "models")
        
        # make directory
        if not os.path.exists(path):
            os.makedirs(path)
        
        f = f'{path}/{name}.pt'
        torch.save(state_dict, f)