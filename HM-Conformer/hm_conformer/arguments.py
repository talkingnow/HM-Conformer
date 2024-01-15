import os
import itertools

def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'project'       : 'ASVspoof2023',
        'name'          : 'HM-Conformer',
        'tags'          : [],
        'description'   : '',

        # log
        'path_log'      : '/results',
        'neptune_user'  : '',
        'neptune_token' : '',
        'wandb_group'   : '',
        'wandb_entity'  : '',
        'wandb_api_key' : '',
        
        # datasets
        'path_train'    : '/data/ASVspoof2019',
        'path_test'     : '/data/ASVspoof2021_DF',
        'path_test_LA'  : None,
        'path_musan'    : '/data/musan',
        'path_rir'      : '/data/RIRS_NOISES/simulated_rirs',

        # others
        'num_workers': 4,
        'usable_gpu': {Available GPUs}, # ex) '0,1'
    }

    experiment_args = {
        'TEST'              : True,
        # experiment
        'epoch'             : 200,
        'batch_size'        : 240,
        'rand_seed'         : 1,
        
        # frontend model
        'bin_size'          : 120,
        'output_size'       : 128,
        'input_layer'       : "conv2d2", 
        'pos_enc_layer_type': "rel_pos",  
        'linear_units'      : 256,
        'cnn_module_kernel' : 15,
        'dropout'           : 0.75,
        'emb_dropout'       : 0.3,
        
        # backend model
        'use_pooling'       : False,
        'input_mean_std'    : False,
        'embedding_size'    : 64,
        
        # OCSoftmax loss
        'num_class'         : 1,
        'feat_dim'          : 2,
        'r_real'            : 0.9,
        'r_fake'            : 0.2,
        'alpha'             : 20.0,
        'loss_weight'       : [0.4, 0.3, 0.2, 0.1, 0.1],
        
        
        # data processing
        'sample_rate'       : 16000, 
        'n_lfcc'            : 40, 
        'coef'              : 0.97, 
        'n_fft'             : 512, 
        'win_length'        : 320, 
        'hop'               : 160, 
        'with_delta'        : True, 
        'with_emphasis'     : True, 
        'with_energy'       : True,
        'train_crop_size'   : 16000 * 4,
        'test_crop_size'    : 16000 * 4,
        
        # data augmentation
        # 1. when Reading file
        'DA_codec_speed'    : True,         # codec: 'aac', 'flac', 'm4a', 'mp3', 'ogg', 'wav', 'wav', 'wma', speed: 'slow', 'fast'
        # 2. when __getitem__
        'DA_p'              : 0.5,
        'DA_list'           : [], # 'ACN': add_coloured_noise, 'FQM': frq_masking, 'MUS': MUSAN, 'RIR': RIR
        'DA_params'         : {
            'MUS': {'path': system_args['path_musan']},
            'RIR': {'path': system_args['path_rir']}  
        },
        # 3. when processing WaveformAugmentation which is in Framework
        'DA_wav_aug_list'   : ['ACN'], 
            # 'ACN': add_colored_noise, 'GAN': gain, 'HPF': high pass filter, 'LPF': low pass filter
            # if use 'HPF' or 'LPF' training speed will be slow
        'DA_wav_aug_params' :  {
            'sr': 16000,
            'ACN': {'min_snr_in_db': 10, 'max_snr_in_db': 40, 'min_f_decay': -2.0, 'max_f_decay': 2.0, 'p': 1},
            'HPF': {'min_cutoff_freq': 20.0, 'max_cutoff_freq': 2400.0, 'p': 0.5},
            'LPF': {'min_cutoff_freq': 150.0, 'max_cutoff_freq': 7500.0, 'p': 0.5},
            'GAN': {'min_gain_in_db': -15.0, 'max_gain_in_db': 5.0, 'p': 0.5}
        },
        # 4. when extracting acoustic_feature
        'DA_frq_p'          : 1,
        'DA_frq_mask'       : True,
        'DA_frq_mask_max'   : 20,
        
        # learning rate
        'lr'                : 1e-6,
        'lr_min'            : 1e-6,
		'weight_decay'      : 1e-4,
        'T_mult'            : 1,
        
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))
    args['path_params'] = args['path_scripts'] + '/params'

    return args, system_args, experiment_args