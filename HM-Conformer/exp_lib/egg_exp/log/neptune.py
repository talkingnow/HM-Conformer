from . import Logger

class NeptuneLogger(Logger):
    """Save experiment logs to neptune
    See here -> https://neptune.ai/
    """
    def __init__(self, user, token, name, project, tags, description=None, scripts=None):
        import neptune.new as neptune
        self.run = neptune.init_run(
            name=name,
            api_token=token,
            project=f"{user}/{project}",
            tags=tags,
            description=description,
            source_files=scripts,
            capture_stderr=False,
            capture_hardware_metrics=False
        )
        
    def log_metric(self, name, value, step=None):
        self.run[name].append(value)

    def log_text(self, name, text):
        if len(text) < 990:
            self.run[name].log(text)
        else:
            strI = text.split('\n')
            for i in range(len(strI)):
                self.run[name].log(strI[i])
        
    def log_image(self, name, image):
        self.run[name].upload(image)

    def log_arguments(self, dictionary):
        for k, v in dictionary.items():
            self.run[f'parameters/{k}'] = v
    
    def save_model(self, name, state_dict):
        pass

    def finish(self):
        pass