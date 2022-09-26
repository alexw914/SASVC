import os
import random
import json
import glob
from pathlib import Path
from speechbrain.dataio.dataio import read_audio
from utils.preprocess_enroll_data import get_enrol_protocols, get_eval_protocols


CM_PROTO_DIR = 'cm/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols'
CM_TRAIN_FILE = 'ASVspoof2019.LA.cm.train.trn.txt'
CM_DEV_FILE = 'ASVspoof2019.LA.cm.dev.trl.txt'
CM_EVAL_FILE = 'ASVspoof2019.LA.cm.eval.trl.txt'
CM_FLAC_DIR = 'cm/ASVspoof2019/LA/ASVspoof2019_LA_%s/flac'
SPK_META_FILE = "spk_meta/spk_meta_%s.pk"

PROCESSED_DATA_DIR = 'spk_meta'

CM_SB_TRAIN_FILE = 'cm_sb_train.json'
CM_SB_DEV_FILE = 'cm_sb_dev.json'
CM_SB_EVAL_FILE = 'cm_sb_eval.json'

SASV_PROTO_DIR = 'cm/ASVspoof2019/LA/ASVspoof2019_LA_asv_protocols'
SASV_DEV_FILE ="ASVspoof2019.LA.asv.dev.gi.trl.txt"
SASV_EVAL_FILE ="ASVspoof2019.LA.asv.eval.gi.trl.txt"

SASV_SB_DEV_FILE = 'sasv_sb_dev.json'
SASV_SB_EVAL_FILE = 'sasv_sb_eval.json'


BONAFIDE = 'bonafide'
SPOOF = 'spoof'

# Statistics
SPOOF_TRAIN_PERCENT = 90
SPOOF_DEV_PERCENT = 10
RANDOM_SEED = 97271
SAMPLERATE = 16000
SPLIT = ['train', 'dev', 'eval']

random.seed(RANDOM_SEED)
Path(PROCESSED_DATA_DIR).mkdir(exist_ok=True)

#Check if we already have SpeechBrain format CM protocol
if os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_TRAIN_FILE)) and \
    os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_DEV_FILE)) and \
        os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_EVAL_FILE)):
    print('SpeechBrain format CM protocols exist...')
else:
    print('Start to convert original CM protocols to SpeechBrain format...')
    # Read CM protocols in train/dev/eval set
    save_files = [CM_SB_TRAIN_FILE, CM_SB_DEV_FILE, CM_SB_EVAL_FILE]
    for i, file in enumerate([CM_TRAIN_FILE, CM_DEV_FILE, CM_EVAL_FILE]):
        cm_features = {}
        with open(os.path.join(CM_PROTO_DIR, file), 'r') as f:
            cm_pros = f.readlines()
            print('% has %d data!'%(file, len(cm_pros)))
        for pro in cm_pros:
            pro = pro.strip('\n').split(' ')
            speaker_id = pro[0]
            auto_file_name = pro[1]
            spoof_id = pro[3]
            bonafide = pro[4]
            cm_features[auto_file_name] = {
                'speaker_id': speaker_id,
                'bonafide': bonafide,
                "spk_meta_path": SPK_META_FILE%(SPLIT[i])
            }
        # Read flac files and durations
        cur_flac_files = glob.glob(os.path.join( CM_FLAC_DIR%(SPLIT[i]),'*.flac'),
                                   recursive=True)

        n_miss = 0
        # Read each utt file and get its duration. Update cm features
        for file in cur_flac_files:
            utt_id = Path(file).stem
            if utt_id in cm_features:
                cm_features[utt_id]['base_dir'] = CM_FLAC_DIR%(SPLIT[i])
            else: n_miss += 1
            

        print('%d files missed description in protocol file in %s set'%(n_miss, SPLIT[i]))
        # Save updated cm features into json
        with open(os.path.join(PROCESSED_DATA_DIR, save_files[i]), 'w') as f:
            json.dump(cm_features, f, indent=2)

#Read SB format CM protocols
with open(os.path.join(PROCESSED_DATA_DIR, CM_SB_TRAIN_FILE), 'r') as f:
    cm_train = json.load(f)
    print('CM train protocols has %d data'%(len(cm_train)))
with open(os.path.join(PROCESSED_DATA_DIR, CM_SB_DEV_FILE), 'r') as f:
    cm_dev = json.load(f)
    print('CM dev protocols has %d data'%(len(cm_dev)))
with open(os.path.join(PROCESSED_DATA_DIR, CM_SB_EVAL_FILE), 'r') as f:
    cm_eval = json.load(f)
    print('CM eval protocols has %d data'%(len(cm_eval)))



SPLIT = ['dev', 'eval']
save_files = [SASV_SB_DEV_FILE,SASV_SB_EVAL_FILE]
for i, file in enumerate([SASV_DEV_FILE, SASV_EVAL_FILE]):
    sasv_features = {}
    with open(os.path.join(SASV_PROTO_DIR, file), 'r') as f:
            sasv_pros = f.readlines()
            print('% has %d data!'%(file, len(sasv_pros)))
    test_index = 0
    for pro in sasv_pros:
            pro = pro.strip('\n').split(' ')
            test_speaker_id = pro[0]
            auto_file_name = "test_list_"+str(test_index)
            test_id = pro[1]
            sasv_label = pro[3]
            sasv_features[auto_file_name] = {
                'test_speaker_id': test_speaker_id,
                'test_id': test_id,
                "spk_meta_path": SPK_META_FILE%(SPLIT[i]),
                "sasv_label": sasv_label, 
                "base_dir": CM_FLAC_DIR%(SPLIT[i])
            }
            test_index = test_index + 1
        # Read flac files and durations
    # cur_flac_files = glob.glob(os.path.join( CM_FLAC_DIR%(SPLIT[i]),'*.flac'),
    #                                recursive=True)

    # n_miss = 0
    #     # Read each utt file and get its duration. Update cm features
    # for file in cur_flac_files:
    #     utt_id = Path(file).stem
    #     if utt_id in cm_features:
    #         sasv_features[utt_id]['base_dir'] = CM_FLAC_DIR%(SPLIT[i])
    #     else: n_miss += 1
            

    # print('%d files missed description in protocol file in %s set'%(n_miss, SPLIT[i]))
        # Save updated cm features into json
    with open(os.path.join(PROCESSED_DATA_DIR, save_files[i]), 'w') as f:
            json.dump(sasv_features, f, indent=2)

speaker_id = set()
for utt_id in cm_train:
    speaker_id.add(cm_train[utt_id]['speaker_id'])
speaker_id = sorted(list(speaker_id))
with open(os.path.join(PROCESSED_DATA_DIR, 'speaker_id_list.json'), 'w') as f:
    json.dump(speaker_id, f)


# ASV_PROTO_DIR = "cm/ASVspoof2019/LA/ASVspoof2019_LA_asv_protocols"
# get_enrol_protocols(pro_dir=ASV_PROTO_DIR, save_dir=PROCESSED_DATA_DIR)
# get_eval_protocols(pro_dir=ASV_PROTO_DIR, save_dir=PROCESSED_DATA_DIR)



