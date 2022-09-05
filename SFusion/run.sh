. path.sh
. cmd.sh

asvspoof_path=/Users/wuyaowang/Database/ASVspoof/LA # add your own path
work_dir=./data/sasv
asv_dir=$work_dir/asv_part
asv_work_dir=$asv_dir/exp
asv_result_dir=$asv_dir/asv_result
asv_emb_dir=$asv_dir/embeddings
cm_dir=$work_dir/cm_part
cm_work_dir=$cm_dir/exp
cm_result_dir=$cm_dir/cm_result
cm_emb_dir=$cm_dir/embeddings

asvspoof_eval_trials=./protocols/ASVspoof2019.LA.cm.eval.trl.txt
asv_eval_trails=./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt

mkdir -p $work_dir || exit 1;
mkdir -p $asv_dir || exit 1;
mkdir -p $asv_work_dir || exit 1;
mkdir -p $asv_result_dir || exit 1;
mkdir -p $asv_emb_dir || exit 1;
mkdir -p $cm_dir || exit 1;
mkdir -p $cm_work_dir || exit 1;
mkdir -p $cm_result_dir || exit 1;
mkdir -p $cm_emb_dir || exit 1;


stage=1
nj=8

# if [ $stage -le 1 ]; then
#     # Prepare the magicdata data
#     echo "Prepare ASVspoof data"
#     scripts/make_asvspoof.sh $asvspoof_path $work_dir
#     utils/utt2spk_to_spk2utt.pl $work_dir/train.utt2spk > $work_dir/train.spk2utt
#     utils/utt2spk_to_spk2utt.pl $work_dir/dev.utt2spk > $work_dir/dev.spk2utt
#     utils/utt2spk_to_spk2utt.pl $work_dir/eval.utt2spk > $work_dir/eval.spk2utt

#     # cp $work_dir/wav.scp $asv_dir
#     # cp $work_dir/train.spk2utt $asv_dir
#     # cp $work_dir/train.utt2spk $asv_dir

#     # utils/fix_data_dir.sh $asv_dir

# fi

# if [ $stage -le 2 ]; then
#     # From pre-train embedding to ark
#     echo "Saving embedding files"
#     system="asv"
#     pre_embd_path=./embd
#     scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "trn"
#     scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "dev"
#     scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "eval"
#     scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "enrol"

# fi


# if [ $stage -le 10 ]; then
# #   Compute the mean vector for centering the evaluation xvectors.

#   $train_cmd $asv_work_dir/xvectors_train/log/compute_mean.log \
#     ivector-mean ark:$asv_emb_dir/embd_asv.trn.ark \
#     $asv_work_dir/xvectors_train/mean.vec || exit 1;

#   # This script uses LDA to decrease the dimensionality prior to PLDA.
#   lda_dim=150
#   $train_cmd $asv_work_dir/xvectors_train/log/lda.log \
#     ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#     ark:$asv_emb_dir/embd_asv.trn.ark \
#     ark:$work_dir/train.utt2spk $asv_work_dir/xvectors_train/transform.mat || exit 1;

#   # Train the PLDA model.
#   $train_cmd $asv_work_dir/xvectors_train/log/plda.log \
#     ivector-compute-plda ark:$work_dir/train.spk2utt \
#     "ark:ivector-subtract-global-mean scp:$asv_emb_dir/embd_asv.trn.scp ark:- | transform-vec $asv_work_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
#     $asv_work_dir/xvectors_train/plda || exit 1;
# fi

# if [ $stage -le 11 ]; then
#   # PLDA scoring

#   $train_cmd $asv_work_dir/scores/log/ASVspoof_eval_scoring.log \
#     ivector-plda-scoring --normalize-length=true \
#     "ivector-copy-plda --smoothing=0.0 $asv_work_dir/xvectors_train/plda - |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/xvectors_train/mean.vec scp:$asv_emb_dir/embd_asv.enrol.scp ark:- | transform-vec $asv_work_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/xvectors_train/mean.vec scp:$asv_emb_dir/embd_asv.eval.scp ark:- | transform-vec $asv_work_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#     "cat '$asvspoof_eval_trials' | awk '{print \\\$1, \\\$2}' |" $asv_work_dir/scores/asvspoof_eval || exit 1;

# fi

if [ $stage -le 12 ]; then
  python scripts/prepare_for_eer.py $asv_eval_trails $asv_work_dir/scores/asvspoof_eval
  python scripts/get_metrics.py --score_file=./asv_score.txt
fi

