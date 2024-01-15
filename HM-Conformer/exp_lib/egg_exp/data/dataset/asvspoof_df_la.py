import os

from ._dataclass import DF_Item

class ASVspoof2021_DF_LA:
    NUM_TEST_ITEM    = 533928
    NUM_TEST_ITEM_LA = 148176

    PATH_TRAIN_TRL  = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    PATH_TRAIN_TRL_DA = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA.txt'
    PATH_TRAIN_TRL_DA_wo_speed  = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_wo_speed.txt'
    
    PATH_DEV_TRL    = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'  
    PATH_DEV_TRL_DA = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_DEV.txt'
    PATH_DEV_TRL_DA_wo_speed    = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_DEV_wo_speed.txt'
    
    PATH_TRAIN_FLAC = 'LA/ASVspoof2019_LA_train'
    PATH_DEV_FLAC   = 'LA/ASVspoof2019_LA_dev'
    
    PATH_TEST_TRL   = 'keys/DF/CM/trial_metadata.txt'
    PATH_TEST_FLAC  = 'ASVspoof2021_DF_eval/flac'
    
    PATH_TEST_TRL_LA = 'keys/LA/CM/trial_metadata.txt'
    PATH_TEST_FLAC_LA = 'flac'
    
    
    def __init__(self, path_train, path_test, path_test_LA=None, use_dev=True, DA=True, DA_speed=True, print_info=False):   
        self.train_set = []
        self.test_set = []
        self.test_set_LA = []
        self.class_weight = []

        # train_set
        train_num_pos = 0
        train_num_neg = 0
        trl = os.path.join(path_train, self.PATH_TRAIN_TRL_DA if DA else self.PATH_TRAIN_TRL)  
        if not DA_speed: 
            trl = os.path.join(path_train, self.PATH_TRAIN_TRL_DA_wo_speed)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')
            if DA:
                f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'{strI[1]}.flac')
            else: 
                f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'flac/{strI[1]}.flac')
            attack_type = strI[3]
            label = 0 if strI[4] == 'bonafide' else 1   # Real: 0, Fake: 1
            if label == 0:
                train_num_neg += 1
            else:
                train_num_pos += 1
            item = DF_Item(f, label, attack_type, is_fake=(label == 1))
            self.train_set.append(item)
            
        # use dev_set in train
        if use_dev:
            trl_dev = os.path.join(path_train, self.PATH_DEV_TRL_DA if DA else self.PATH_DEV_TRL)
            if not DA_speed: 
                trl_dev = os.path.join(path_train, self.PATH_DEV_TRL_DA_wo_speed)
            for line in open(trl_dev).readlines():
                strI = line.replace('\n', '').split(' ')
                if DA:
                    f = os.path.join(path_train, self.PATH_DEV_FLAC, f'{strI[1]}.flac')
                else: 
                    f = os.path.join(path_train, self.PATH_DEV_FLAC, f'flac/{strI[1]}.flac')
                attack_type = strI[3]
                label = 0 if strI[4] == 'bonafide' else 1   # Real: 0, Fake: 1
                if label == 0:
                    train_num_neg += 1
                else:
                    train_num_pos += 1
                item = DF_Item(f, label, attack_type, is_fake=(label == 1))
                self.train_set.append(item)
        
        self.class_weight.append((train_num_neg + train_num_pos) / train_num_neg)
        self.class_weight.append((train_num_neg + train_num_pos) / train_num_pos)
        
        # test_set
        test_num_pos = 0
        test_num_neg = 0
        trl = os.path.join(path_test, self.PATH_TEST_TRL)
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')
            # check subset
            if strI[7] != 'eval':
                continue
            f = os.path.join(path_test, self.PATH_TEST_FLAC, f'{strI[1]}.flac')
            attack_type = strI[4]
            label = 0 if attack_type == '-' else 1
            if label == 0:
                test_num_neg += 1
            else:
                test_num_pos += 1
                
            item = DF_Item(f, label, attack_type, is_fake=label == 1)
            self.test_set.append(item)
        # error check
        assert len(self.test_set) == self.NUM_TEST_ITEM, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set)}, EXPECTED: {self.NUM_TEST_ITEM}'

        # test_set_LA
        if path_test_LA is not None:
            test_num_pos_LA = 0
            test_num_neg_LA = 0
            trl_LA = os.path.join(path_test_LA, self.PATH_TEST_TRL_LA)
            for line in open(trl_LA).readlines():
                strI = line.replace('\n', '').split(' ')
                # check subset
                if strI[7] != 'eval':
                    continue
                f = os.path.join(path_test_LA, self.PATH_TEST_FLAC_LA, f'{strI[1]}.flac')
                attack_type = strI[4]
                label = 0 if attack_type == 'bonafide' else 1
                if label == 0:
                    test_num_neg_LA += 1
                else:
                    test_num_pos_LA += 1
                    
                item = DF_Item(f, label, attack_type, is_fake=label == 1)
                self.test_set_LA.append(item)
            # error check
            assert len(self.test_set_LA) == self.NUM_TEST_ITEM_LA, f'[DATASET ERROR] - TEST_SAMPLE: {len(self.test_set_LA)}, EXPECTED: {self.NUM_TEST_ITEM_LA}'
        
        # print info
        if path_test_LA is not None and print_info:
                info = (
                    f'====================\n'
                    + f'    ASVspoof2021    \n'
                    + f'====================\n'
                    + f'TRAIN (ASVspoof2019 LA): bona - {train_num_neg}, spoof - {train_num_pos}\n'
                    + f'TEST  (ASVspoof2021 DF):  bona - {test_num_neg}, spoof - {test_num_pos}\n'
                    + f'TEST  (ASVspoof2021 LA):  bona - {test_num_neg_LA}, spoof - {test_num_pos_LA}\n'
                    + f'====================\n'
                )
                print(info)
        else:
            if print_info:
                info = (
                    f'====================\n'
                    + f'    ASVspoof2021    \n'
                    + f'====================\n'
                    + f'TRAIN (ASVspoof2019 LA): bona - {train_num_neg}, spoof - {train_num_pos}\n'
                    + f'TEST  (ASVspoof2021 DF):  bona - {test_num_neg}, spoof - {test_num_pos}\n'
                    + f'====================\n'
                )
                print(info)