. path.sh
. cmd.sh

asvspoof_path=/home/alex/Corpora/ASVspoof2019/LA # add your own path
work_dir=./data/sasv
asv_dir=$work_dir/asv_part
asv_work_dir=$asv_dir/exp
asv_result_dir=$asv_dir/asv_result
asv_emb_dir=$asv_dir/embeddings
whiten_asv_emb_dir=$asv_dir/whiten_embeddings
cm_dir=$work_dir/cm_part
cm_work_dir=$cm_dir/exp
cm_result_dir=$cm_dir/cm_result
cm_emb_dir=$cm_dir/embeddings

asvspoof_eval_trials=./protocols/ASVspoof2019.LA.cm.eval.trl.txt
asv_eval_trails=./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt
vox1_path=/home/alex/dataset/VoxCeleb/VoxCeleb1
vox2_path=/home/alex/dataset/VoxCeleb/VoxCeleb2

mkdir -p $work_dir || exit 1;
mkdir -p $asv_dir || exit 1;
mkdir -p $asv_work_dir || exit 1;
mkdir -p $asv_result_dir || exit 1;
mkdir -p $asv_emb_dir || exit 1;
mkdir -p $cm_dir || exit 1;
mkdir -p $cm_work_dir || exit 1;
mkdir -p $cm_result_dir || exit 1;
mkdir -p $cm_emb_dir || exit 1;


stage=5
nj=8

if [ $stage -le 1 ]; then
    # Prepare the magicdata data
    echo "Prepare ASVspoof data"
    scripts/make_asvspoof.sh $asvspoof_path $work_dir
    utils/utt2spk_to_spk2utt.pl $work_dir/train/utt2spk > $work_dir/train/spk2utt
    utils/utt2spk_to_spk2utt.pl $work_dir/dev/utt2spk > $work_dir/dev/spk2utt
    utils/utt2spk_to_spk2utt.pl $work_dir/eval/utt2spk > $work_dir/eval/spk2utt
    
    local/make_voxceleb1_v2.pl $vox1_path dev data/vox1
    local/make_voxceleb1_v2.pl $vox2_path dev data/vox2      
    utils/combine_data.sh data/vox data/vox1 data/vox2
    utils/combine_data.sh data/trn $work_dir/train $work_dir/dev
    # utils/fix_data_dir.sh $asv_dir
    rm -R data/vox1
    rm -R data/vox2

fi

if [ $stage -le 2 ]; then
    # From pre-train embedding to ark
    echo "Saving embedding files"
    system="asv"

    pre_embd_path=./embd_ecapa
    
    scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "train" 
    scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "eval"
    scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "enrol"
    system="vox_asv"
    scripts/save_embeddings.sh  $pre_embd_path $system $asv_dir "dev"
fi

# compute mean vector and using LDA
# if [ $stage -le 3 ]; then

#   # Compute the mean vector for voxceleb.
#   $train_cmd $asv_work_dir/vox_plda/log/compute_mean.log \
#     ivector-mean ark:$asv_emb_dir/embd_vox_asv.dev.ark \
#     $asv_work_dir/vox_plda/mean.vec || exit 1;
#   # Compute the mean vector for asvspoof.
#   $train_cmd $asv_work_dir/asvspoof_plda/log/compute_mean.log \
#     ivector-mean ark:$asv_emb_dir/embd_asv.train.ark \
#     $asv_work_dir/asvspoof_plda/mean.vec || exit 1;
 
#   # This script uses LDA to decrease the dimensionality prior to PLDA.
#   lda_dim=150
#   $train_cmd $asv_work_dir/vox_plda/log/lda.log \
#     ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#     ark:$asv_emb_dir/embd_vox_asv.dev.ark \
#     ark:data/vox/utt2spk $asv_work_dir/vox_plda/transform.mat || exit 1;
#   $train_cmd $asv_work_dir/asvspoof_plda/log/lda.log \
#     ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#     ark:$asv_emb_dir/embd_asv.train.ark \
#     ark:data/trn/utt2spk $asv_work_dir/asvspoof_plda/transform.mat || exit 1;

# fi


if [ $stage -le 4 ]; then

  # compute cosine distance
  cat $asv_eval_trails | cut -d\  --fields=1,2 > $asv_work_dir/trails | exit 1


  # warning: ivector-compute-cosine need change the kaldi code and compile， if you want compute cosine similarty use ark file.

  ivector-compute-cosine $asv_work_dir/trails ark:$asv_emb_dir/embd_asv.enrol.ark ark:$asv_emb_dir/embd_asv.eval.ark $asv_result_dir/asv_cosine.txt

  awk '{print $3}' $asv_result_dir/asv_cosine.txt | paste - $asv_eval_trails | awk '{print $1, $5}' > $asv_result_dir/score_cosine.txt
  python local/get_sasv_metrics.py --score_file=$asv_result_dir/score_cosine.txt

fi

# PLDA using LDA
# if [ $stage -le 6 ]; then

#   $train_cmd $asv_work_dir/vox_plda/log/plda.log \
#     subtools/score/pyplda/test_PLDA.sh data/vox/spk2utt \
#     "ark:ivector-subtract-global-mean ark:$asv_emb_dir/embd_vox_asv.dev.scp ark:- | transform-vec $asv_work_dir/vox_plda/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/vox_plda/mean.vec scp:$asv_emb_dir/embd_asv.enrol.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/vox_plda/mean.vec scp:$asv_emb_dir/embd_asv.eval.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#     $asv_work_dir/vox_plda/plda \
#     $asv_eval_trails $asv_result_dir/asv_eval.txt;

#   $train_cmd $asv_work_dir/asvspoof_plda/log/plda.log \
#     subtools/score/pyplda/test_PLDA.sh data/trn/spk2utt \
#     "ark:ivector-subtract-global-mean ark:$asv_emb_dir/embd_asv_train.dev.scp ark:- | transform-vec $asv_work_dir/asvspoof_plda/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/asvspoof_plda/mean.vec scp:$asv_emb_dir/embd_asv.enrol.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/asvspoof_plda/mean.vec scp:$asv_emb_dir/embd_asv.eval.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#     $asv_work_dir/asvspoof_plda/plda \
#     $asv_eval_trails $asv_result_dir/asv_eval.txt;  

# fi

if [ $stage -le 7 ]; then

  # $train_cmd $asv_work_dir/vox_plda/log/plda.log \
  #     subtools/score/pyplda/test_PLDA.sh data/vox/spk2utt \
  #     ark:$asv_emb_dir/embd_vox_asv.dev.ark \
  #     ark:$asv_emb_dir/embd_asv.enrol.ark\
  #     ark:$asv_emb_dir/embd_asv.eval.ark\
  #     $asv_work_dir/vox_plda/plda \
  #     $asv_eval_trails $asv_result_dir/asv_eval.txt;

  $train_cmd $asv_work_dir/asvspoof_plda/log/plda.log \
      subtools/score/pyplda/test_PLDA.sh data/trn/spk2utt \
      ark:$asv_emb_dir/embd_asv.train.ark \
      ark:$asv_emb_dir/embd_asv.enrol.ark\
      ark:$asv_emb_dir/embd_asv.eval.ark\
      $asv_work_dir/asvspoof_plda/plda \
      $asv_eval_trails $asv_result_dir/asv_eval.txt;   

fi  

# if [ $stage -le 7 ]; then

#   # PLDA scoring
#   $train_cmd $asv_work_dir/vox_plda/log/plda.log \
#     subtools/score/pyplda/test_PLDA.sh data/vox/spk2utt \
#     ark:$asv_emb_dir/embd_vox_asv.dev.ark \
#     "ark:ivector-subtract-global-mean $asv_work_dir/vox_plda/mean.vec scp:$asv_emb_dir/embd_asv.enrol.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#     "ark:ivector-subtract-global-mean $asv_work_dir/vox_plda/mean.vec scp:$asv_emb_dir/embd_asv.eval.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#     $asv_work_dir/vox_plda/plda \
#     $asv_eval_trails $asv_result_dir/asv_eval.txt;

#   # # PLDA scoring
#   # $train_cmd $asv_work_dir/asvspoof_plda/log/plda.log \
#   #   subtools/score/pyplda/test_PLDA.sh data/trn/spk2utt \
#   #   ark:$asv_emb_dir/embd_asv.train.ark \
#   #   "ark:ivector-subtract-global-mean $asv_work_dir/asvspoof_plda/mean.vec scp:$asv_emb_dir/embd_asv.enrol.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#   #   "ark:ivector-subtract-global-mean $asv_work_dir/asvspoof_plda/mean.vec scp:$asv_emb_dir/embd_asv.eval.scp ark:- | ivector-normalize-length ark:- ark:- |" \
#   #   $asv_work_dir/asvspoof_plda/plda \
#   #   $asv_eval_trails $asv_result_dir/asv_eval.txt;     

# fi

# if [ $stage -le 8 ]; then

#   # APLDA
#   $train_cmd $asv_work_dir/asvspoof_plda/log/aplda.log \
#     subtools/score/pyplda/test_APLDA.sh $asv_work_dir/vox_plda/plda.ori \
#     ark:$asv_emb_dir/embd_asv.train.ark\
#     ark:$asv_emb_dir/embd_asv.enrol.ark\
#     ark:$asv_emb_dir/embd_asv.eval.ark\
#     $asv_work_dir/asvspoof_plda/plda.adapt\
#     $asv_eval_trails \
#     $asv_result_dir/asv_eval.txt;    

# fi

# if [ $stage -le 8 ]; then

#   # # CORAL
#   # $train_cmd $asv_work_dir/asvspoof_plda/log/coral.log \
#   #   subtools/score/pyplda/test_CORAL.sh $asv_work_dir/vox_plda/plda.ori \
#   #   ark:$asv_emb_dir/embd_asv.train.ark\
#   #   ark:$asv_emb_dir/embd_asv.enrol.ark\
#   #   ark:$asv_emb_dir/embd_asv.eval.ark\
#   #   $asv_work_dir/asvspoof_plda/plda.adapt\
#   #   $asv_eval_trails \
#   #   $asv_result_dir/asv_eval.txt;  

#   # CORAL PLUS
#   $train_cmd $asv_work_dir/asvspoof_plda/log/coralplus.log \
#     subtools/score/pyplda/test_CORAL_Plus.sh $asv_work_dir/vox_plda/plda.ori \
#     ark:$asv_emb_dir/embd_asv.train.ark\
#     ark:$asv_emb_dir/embd_asv.enrol.ark\
#     ark:$asv_emb_dir/embd_asv.eval.ark\
#     $asv_work_dir/asvspoof_plda/plda.adapt\
#     $asv_eval_trails \
#     $asv_result_dir/asv_eval.txt;

# fi

# if [ $stage -le 9 ];then

#   # # CIP
#   # $train_cmd $asv_work_dir/asvspoof_plda/log/cip.log \
#   #   subtools/score/pyplda/test_CIP.sh $asv_work_dir/vox_plda/plda.ori \
#   #   ark:$asv_emb_dir/embd_asv.train.ark\
#   #   $asv_work_dir/asvspoof_plda/plda.ori\
#   #   ark:$asv_emb_dir/embd_asv.enrol.ark\
#   #   ark:$asv_emb_dir/embd_asv.eval.ark\
#   #   $asv_work_dir/asvspoof_plda/plda.adapt\
#   #   $asv_eval_trails \
#   #   $asv_result_dir/asv_eval.txt;
  
#   # CIP_Reg
#   $train_cmd $asv_work_dir/asvspoof_plda/log/cip_reg.log \
#     subtools/score/pyplda/test_CIP_Reg.sh $asv_work_dir/vox_plda/plda.ori \
#     ark:$asv_emb_dir/embd_asv.train.ark\
#     $asv_work_dir/asvspoof_plda/plda.ori\
#     ark:$asv_emb_dir/embd_asv.enrol.ark\
#     ark:$asv_emb_dir/embd_asv.eval.ark\
#     $asv_work_dir/asvspoof_plda/plda.adapt\
#     $asv_eval_trails \
#     $asv_result_dir/asv_eval.txt; 

# fi

# if [ $stage -le 10 ];then

#   # # LIP
#   # $train_cmd $asv_work_dir/asvspoof_plda/log/lip.log \
#   #   subtools/score/pyplda/test_LIP.sh $asv_work_dir/vox_plda/plda.ori \
#   #   $asv_work_dir/asvspoof_plda/plda.ori\
#   #   ark:$asv_emb_dir/embd_asv.enrol.ark\
#   #   ark:$asv_emb_dir/embd_asv.eval.ark\
#   #   $asv_work_dir/asvspoof_plda/plda.adapt\
#   #   $asv_eval_trails \
#   #   $asv_result_dir/asv_eval.txt;

#   # LIP_Reg
#   $train_cmd $asv_work_dir/asvspoof_plda/log/lip_reg.log \
#     subtools/score/pyplda/test_LIP_Reg.sh $asv_work_dir/vox_plda/plda.ori \
#     $asv_work_dir/asvspoof_plda/plda.ori\
#     ark:$asv_emb_dir/embd_asv.enrol.ark\
#     ark:$asv_emb_dir/embd_asv.eval.ark\
#     $asv_work_dir/asvspoof_plda/plda.adapt\
#     $asv_eval_trails \
#     $asv_result_dir/asv_eval.txt;

# fi


if [ $stage -le 12 ]; then

  python local/prepare_sasv_eer.py $asv_eval_trails $asv_result_dir/asv_eval.txt $cm_result_dir/cm_eval.txt $work_dir/sasv_score.txt;
  python local/get_sasv_metrics.py --score_file=$work_dir/sasv_score.txt

fi

