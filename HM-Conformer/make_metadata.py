import os

def read_metadata_ASVspoof2019(path_meta):
    """
    Read metadata -> dictionary
    key: file_name, items: infomation
    """
    metadata = {}
    cnt = 0
    for line in open(path_meta).readlines():
        strI = line.replace('\n', '').split(' ')
        metadata[strI[1]] = line
        cnt += 1
    print(f'lines: {cnt}')
    return metadata

def write_DA_metadata(
        org_metadata,
        path,
        option,
        exception=[],
        codecs=['flac']
    ):
    if option == 'dev':
        path_train = path + '/LA/ASVspoof2019_LA_dev',
        path_write = path + '/LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA_DEV.txt',
    elif option == 'trn':
        path_train = path + '/LA/ASVspoof2019_LA_train',
        path_write = path + '/LA/ASVspoof2019_LA_cm_protocols/metadata_with_DA.txt',
    with open(path_write, 'w') as f:
        for codec in codecs:
            _path_train = path_train + "/" + codec
            for root, _, files in os.walk(_path_train):
                for file in files:
                    if '.flac' in file:
                        # if '_fast' in file or '_slow' in file: 
                        #     continue
                        f_dir = root.split('/')[-1]     # flac
                        f_name = file.split('.')[0]     # LA_T_*
                        if f_name in exception:
                            print('exception file')
                            continue
                        org_name = f_name[:12]  # LA_T_0000000
                        
                        if org_name[:6] == 'LA_D_A':
                            new_file = f_dir + '/' + f_name
                            new_line = f"LA_0000 {new_file} - - bonafide\n"
                        else:
                            line = org_metadata[org_name]
                            new_line = line.replace(org_name, f_dir + '/' + f_name)
                        f.write(new_line)

        
if __name__ == '__main__':
    YOUR_ASVspoof2019_PATH = {YOUR_ASVspoof2019_PATH}   # '/home/shin/exps/DB/ASVspoof2019'
    path_meta_trn = YOUR_ASVspoof2019_PATH + '/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    metadata = read_metadata_ASVspoof2019(path_meta_trn)
    write_DA_metadata(metadata, YOUR_ASVspoof2019_PATH, 'trn')

    path_meta_dev = YOUR_ASVspoof2019_PATH + '/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    metadata = read_metadata_ASVspoof2019(path_meta_dev)
    write_DA_metadata(metadata, YOUR_ASVspoof2019_PATH, 'dev')