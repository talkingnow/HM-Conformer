import os
import sys
import random
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

if os.path.exists('/exp_lib'):
    sys.path.append('/exp_lib')
import egg_exp
import arguments
import data_processing
import train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def run(process_id, args, experiment_args):
    #===================================================
    #                    Setting      
    #===================================================
    torch.cuda.empty_cache()
    
    # set reproducible
    set_seed(args['rand_seed'])
    
    # DDP 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = process_id
    args['device'] = f'cuda:{process_id}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])
    flag_parent = process_id == 0

    # logger
    if flag_parent:
        builder = egg_exp.log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
        builder.use_local_logger(args['path_log'])
        # builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
        # builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
        logger = builder.build()
        logger.log_arguments(experiment_args)
    else:
        logger = None
    
    # data loader
    asvspoof = egg_exp.data.dataset.ASVspoof2021_DF_LA(args['path_train'], args['path_test'], args['path_test_LA'], args['DA_codec_speed'], print_info=flag_parent)
    
    train_set = data_processing.TrainSet(asvspoof.train_set, args['train_crop_size'], args['DA_p'], args['DA_list'], args['DA_params'])
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoader(
        train_set,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    test_set_DF = data_processing.TestSet(asvspoof.test_set, args['test_crop_size'])
    test_sampler = DistributedSampler(test_set_DF, shuffle=False)
    test_loader_DF = DataLoader(
        test_set_DF,
        num_workers=args['num_workers'],
        batch_size=args['batch_size'],
        pin_memory=True,
        sampler=test_sampler,
        drop_last=False
    )
    
    # Waveform augmentation
    augmentation = None
    if len(args['DA_wav_aug_list']) != 0:
        augmentation = egg_exp.data.augmentation.WaveformAugmetation(args['DA_wav_aug_list'], args['DA_wav_aug_params'])
    
    # data preprocessing
    preprocessing = egg_exp.framework.model.LFCC(args['sample_rate'], args['n_lfcc'], 
            args['coef'], args['n_fft'], args['win_length'], args['hop'], args['with_delta'], args['with_emphasis'], args['with_energy'],
            args['DA_frq_mask'], args['DA_frq_p'], args['DA_frq_mask_max'])
 
    # frontend
    frontend = egg_exp.framework.model.HM_Conformer(bin_size=args['bin_size'], output_size=args['output_size'], input_layer=args['input_layer'],
            pos_enc_layer_type=args['pos_enc_layer_type'], linear_units=args['linear_units'], cnn_module_kernel=args['cnn_module_kernel'],
            dropout=args['dropout'], emb_dropout=args['emb_dropout'], multiloss=True)

    # backend
    backends = []
    criterions = []
    for i in range(5):
        backend = egg_exp.framework.model.CLSBackend(in_dim=args['output_size'], hidden_dim=args['embedding_size'], use_pooling=args['use_pooling'], input_mean_std=args['input_mean_std'])
        backends.append(backend)
        
        # criterion
        criterion = egg_exp.framework.loss.OCSoftmax(embedding_size=args['embedding_size'], 
            num_class=args['num_class'], feat_dim=args['feat_dim'], r_real=args['r_real'], 
            r_fake=args['r_fake'], alpha=args['alpha'])
        criterions.append(criterion)
    
    # set framework
    if augmentation != None:
        framework = egg_exp.framework.DeepfakeDetectionFramework_DA_multiloss(
            augmentation=augmentation,
            preprocessing=preprocessing,
            frontend=frontend,
            backend=backends,
            loss=criterions,
            loss_weight=args['loss_weight'],
        )
    else:
        framework = egg_exp.framework.DeepfakeDetectionFramework(
            preprocessing=preprocessing,
            frontend=frontend,
            backend=backend,
            loss=criterion,
        )
    framework.use_distributed_data_parallel(f'cuda:{process_id}', True)

    # optimizer
    optimizer = torch.optim.Adam(framework.get_parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args['epoch'],
        T_mult=args['T_mult'],
        eta_min=args['lr_min']
    )

    # ===================================================
    #                    Test
    # ===================================================
    if args['TEST']:
        framework.load_model(args)
        eer = train.test(framework, test_loader_DF)
        if logger is not None:
            logger.log_metric('DF_EER', eer, 0)
            print('DF: ',eer)

    # ===================================================
    #                    Train
    # ===================================================
    else:
        best_eer_DF = 100
        best_state_DF = framework.copy_state_dict()
        cnt_early_stop = 0

        # load model
        pre_trained_model = os.path.join(args['path_scripts'], 'model')
        if os.path.exists(pre_trained_model):
            state_dict = {}
            for pt in os.listdir((pre_trained_model)):
                state_dict[pt.replace('.pt', '')] = torch.load(pt)
            framework.load_state_dict(state_dict)

        for epoch in range(1, args['epoch'] + 1):
            scheduler.step(epoch)

            # train
            train_sampler.set_epoch(epoch)
            train.train(epoch, framework, optimizer, train_loader, logger)

            # test_DF
            if epoch % 5 == 0:
                cnt_early_stop += 1
                eer = train.test(framework, test_loader_DF)

                # logging
                if eer < best_eer_DF:
                    cnt_early_stop = 0
                    best_eer_DF = eer
                    best_state_ft = framework.copy_state_dict()
                    if logger is not None:
                        logger.log_metric('BestEER', eer, epoch)
                        for key, v in best_state_ft.items():
                            logger.save_model(
                                f'check_point_DF_{key}_{epoch}', v)
                if logger is not None:
                    logger.log_metric('EER', eer, epoch)
                if cnt_early_stop >= 6:
                    break
                

if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    # set reproducible
    set_seed(args['rand_seed'])

    # check gpu environment
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'
    
    args['port'] = f'10{datetime.datetime.now().microsecond % 100}'

    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    args['batch_size'] = args['batch_size'] // args['world_size']
    if args['batch_size'] % args['world_size'] != 0:
        print(f'The batch size is resized to {args["batch_size"] * args["world_size"]} because the rest are discarded.')
    torch.cuda.empty_cache()
    
    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args)
    )