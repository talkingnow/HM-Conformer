import torch
from .interface import Framework

class DeepfakeDetectionFramework_DA_multiloss(Framework):
    def __init__(self, augmentation, preprocessing, frontend, backend, loss, loss_weight):
        super(DeepfakeDetectionFramework_DA_multiloss, self).__init__()

        self.num_loss = len(loss_weight)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.trainable_modules['frontend'] = frontend
        for i in range(self.num_loss):
            self.trainable_modules[f'backend{i}'] = backend[i]
            self.trainable_modules[f'loss{i}'] = loss[i]
        self.loss_weight = loss_weight

    def __call__(self, x, label=None, all_loss=False):
        # pre_processing
        with torch.set_grad_enabled(False):
            x = self.augmentation(x)
        x = self.preprocessing(x)

        # feed forward
        x, embedding = self.trainable_modules['frontend'](x)
        embed_list = []
        for i in range(self.num_loss-1):
            embed_list.append(self.trainable_modules[f'backend{i}'](embedding[:, i, :].unsqueeze(1)))
        embed_list.append(self.trainable_modules[f'backend{self.num_loss-1}'](x))
            
        # loss 
        if label is not None:
            loss_embs = []
            final_loss = []
            for i in range(self.num_loss):
                loss_emb = self.trainable_modules[f'loss{i}'](embed_list[i], label)
                loss_embs.append(loss_emb)
                final_loss.append(loss_emb * self.loss_weight[i])
            
            final_loss = sum(final_loss)
            return x, final_loss, loss_embs
        else:
            if all_loss:
                loss_embs=[]
                for i in range(self.num_loss):
                    loss_emb = self.trainable_modules[f'loss{i}'](embed_list[i], label)
                    loss_embs.append(loss_emb)
                return loss_embs
            x = self.trainable_modules[f'loss{self.num_loss-1}'](embed_list[-1])
            return x
        
    def load_model(self, args):
        path = args['path_params'] + '/'
        param_list = ['frontend',
                      'backend0',
                      'backend1',
                      'backend2',
                      'backend3',
                      'backend4',
                      'loss0',
                      'loss1',
                      'loss2',
                      'loss3',
                      'loss4',]
        load_param_list = ['check_point_DF_frontend_20.pt',
                      'check_point_DF_backend0_20.pt',
                      'check_point_DF_backend1_20.pt',
                      'check_point_DF_backend2_20.pt',
                      'check_point_DF_backend3_20.pt',
                      'check_point_DF_backend4_20.pt',
                      'check_point_DF_loss0_20.pt',
                      'check_point_DF_loss1_20.pt',
                      'check_point_DF_loss2_20.pt',
                      'check_point_DF_loss3_20.pt',
                      'check_point_DF_loss4_20.pt',]
        save_path = '/code/final_test/test/params'
        for i in range(11):
            self.set_params(param_list[i], path+load_param_list[i], save_path, load_param_list[i])
        
    def set_params(self, module_name, path, pp, p):
        self_state = self.trainable_modules[module_name].state_dict()
        loaded_state = torch.load(path, map_location=torch.device('cpu'))

        # with open(f'{pp}/{module_name}.txt', 'w') as f:
        #     for key, model in self_state.items():
        #         f.write(key + '\n')
        # with open(f'{pp}/{p}.txt', 'w') as f:
        #     for key, model in loaded_state.items():
        #         f.write(key + '\n')

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state.keys():
                name = name[7:] # remove "module."
                if name not in self_state.keys():
                    print(f"{origname} is not in the model. ")
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)