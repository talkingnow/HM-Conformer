import os
import time
import shutil
import random
import argparse
import subprocess
# umask 000
class ExperimentSocket:
    def __init__(self, gpu, project, path_root):
        path_root += f'/{project}'
        self.path_queue_task = f'{path_root}/task'
        self.path_queue_current = f'{path_root}/GPU{gpu}/current'

        # set available gpu
        self.num_gpu = 1
        for ch in gpu:
            if(ch == ','):
                self.num_gpu += 1
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # init path
        if os.path.exists(self.path_queue_current):
            shutil.rmtree(self.path_queue_current)
        os.makedirs(self.path_queue_task, exist_ok=True)
        os.makedirs(self.path_queue_current, exist_ok=True)
        
    def run(self):
        while(True):
            experiment = self.get_experiment()

            if experiment is None:
                t = random.randint(5, 10)
                time.sleep(t)
                continue
            else:
                os.system('clear')
                print('Ready: ', experiment)
                t = random.randint(5, 10)
                time.sleep(t)

                if os.path.exists(f'{self.path_queue_task}/{experiment}'):
                    shutil.move(
                        f'{self.path_queue_task}/{experiment}', 
                        f'{self.path_queue_current}/{experiment}'
                    )

                    try:
                        print('Start: ', experiment)
                        subprocess.check_call([
                            'python', 
                            f"{self.path_queue_current}/{experiment}/main.py"
                        ])
                        shutil.rmtree(f'{self.path_queue_current}/{experiment}')

                    except:
                        if os.path.exists(f'{self.path_queue_current}/{experiment}'):
                            shutil.rmtree(f'{self.path_queue_current}/{experiment}')

    def get_experiment(self):
        for dir in os.listdir(self.path_queue_task):
            if dir[0] != '.':
                return dir
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=True)
    parser.add_argument('-project', type=str, required=True)
    args = parser.parse_args()

    experiment_socket = ExperimentSocket(
        args.gpu, 
        args.project,
        os.path.dirname(os.path.realpath(__file__))
    )

    experiment_socket.run()