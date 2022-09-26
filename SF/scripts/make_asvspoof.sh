set -e
asvspoof_path=$1
work_path=$2


python scripts/make_asvspoof.py --init_wav_path $asvspoof_path/ASVspoof2019_LA_train/flac --init_protocol_path $asvspoof_path/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt --init_save_path $work_path/train/ 
python scripts/make_asvspoof.py --init_wav_path $asvspoof_path/ASVspoof2019_LA_dev/flac --init_protocol_path $asvspoof_path/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt --init_save_path $work_path/dev/ 
python scripts/make_asvspoof.py --init_wav_path $asvspoof_path/ASVspoof2019_LA_eval/flac --init_protocol_path $asvspoof_path/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --init_save_path $work_path/eval/ 
