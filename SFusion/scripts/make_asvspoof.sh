set -e
asvspoof_path=$1
work_path=$2


python scripts/make_asvspoof.py --init_wav_path $asvspoof_path/ASVspoof2019_LA_train/flac --init_wav_scp_file $work_path/wav.scp --init_protocol_path $asvspoof_path/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --init_utt2spk_path $work_path/train.utt2spk 
