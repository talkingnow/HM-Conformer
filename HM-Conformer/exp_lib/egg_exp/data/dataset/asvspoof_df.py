import os

from ._dataclass import DF_Item

class ASVspoof2021_DF:
    NUM_TEST_ITEM   = 533928

    PATH_TRAIN_TRL  = 'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    # HS fix here
    PATH_TRAIN_TRL_DA = 'LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA.txt'  # HS
    PATH_TRAIN_FLAC = 'LA/ASVspoof2019_LA_train'
    PATH_TEST_TRL   = 'keys/DF/CM/trial_metadata.txt'
    PATH_TEST_FLAC  = 'ASVspoof2021_DF_eval/flac'

    # HS fix here
    def __init__(self, path_train, path_test, DA=True, print_info=False):   # HS
        self.train_set = []
        self.test_set = []
        self.class_weight = []

        # train_set
        train_num_pos = 0
        train_num_neg = 0
        # HS fix here
        trl = os.path.join(path_train, self.PATH_TRAIN_TRL_DA if DA else self.PATH_TRAIN_TRL)   # HS
        for line in open(trl).readlines():
            strI = line.replace('\n', '').split(' ')

            f = os.path.join(path_train, self.PATH_TRAIN_FLAC, f'{strI[1]}.flac')
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

        # print info
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