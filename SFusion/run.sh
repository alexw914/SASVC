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
    
#     cp $work_dir/wav.scp $asv_dir
#     cp $work_dir/utt2spk $asv_dir
#     cp $work_dir/spk2utt $asv_dir
#     cp $work_dir/train.spk2utt $asv_dir
#     cp $work_dir/train.utt2spk $asv_dir

#     utils/fix_data_dir.sh $asv_dir

# fi

# if [ $stage -le 2 ]; then
#     # From pre-train embedding to ark
#     echo "Saving embedding files"
#     phase="trn"
#     system="asv"
#     pre_embd_path=./embd
#     scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir $phase

# fi


if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.

  # $train_cmd $asv_work_dir/xvectors_train/log/compute_mean.log \
  #   ivector-mean ark:$asv_emb_dir/embd_asv.trn.ark \
  #   $asv_work_dir/xvectors_train/mean.vec || exit 1;

  # # This script uses LDA to decrease the dimensionality prior to PLDA.
  # lda_dim=150
  # $train_cmd $asv_work_dir/xvectors_train/log/lda.log \
  #   ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
  #   ark:$asv_emb_dir/embd_asv.trn.ark \
  #   ark:$asv_dir/train.utt2spk $asv_work_dir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $asv_work_dir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:$asv_dir/train.spk2utt \
    'ark:ivector-normalize-length scp:./data/sasv/asv_part/embeddings/embd_asv.trn.scp ark:./data/sasv/asv_part/embeddings/embd_asv.trn.ark |' \
    exp/ivector_train_1024/plda
fi

# if [ $stage -le 11 ]; then
#   $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
#     ivector-plda-scoring --normalize-length=true \
#     "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
#     "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#     "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#     "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
# fi

# if [ $stage -le 12 ]; then
#   eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
#   mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
#   mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
#   echo "EER: $eer%"
#   echo "minDCF(p-target=0.01): $mindcf1"
#   echo "minDCF(p-target=0.001): $mindcf2"
#   # EER: 3.128%
#   # minDCF(p-target=0.01): 0.3258
#   # minDCF(p-target=0.001): 0.5003
#   #
#   # For reference, here's the ivector system from ../v1:
#   # EER: 5.329%
#   # minDCF(p-target=0.01): 0.4933
#   # minDCF(p-target=0.001): 0.6168
# fi

