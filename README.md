# SASVC
Spoofing-Aware Speaker Verification

This project provide three ways to realize Sooofing-aware Speaker Verification system. Score Fusion, embedding fusion and muti-task learning.

## Score Fusion
It takes PLDA backend as classifier, in this way. Unsupervised domain adaptation and supervised domain adaptation were applied in score fusion method to improve the speaker verification performance. It needs kaldi and asv-subtools. When using pretrained model of ECAPA-TDNN and LIP-Reg adaptation, it gets best Speaker Verification EER of 1.47%. Countermeasure score produced by Wav2Vec-AASIST EER was 0.20%. The sasv score is the multiplication of asv score and cm score processed by sigmoid function. The final  results on eval set is 
SASV: 1.06%, SV: 1.53%, SPF: 0.64%.

## Embedding Fusion
Using Conv1D layer and SEModule to train a SASV model using embeddings from pre-trained asv system and countermeature system provided by SASV baseline system. In eval set, it get results of
SASV: 0.96% SV: 1.24% SPF: 0.68%.

## Multi-task
In this method, i used a pretrained asv system and Attentive statistic pooling layers and fusion backend to build a SASV model. In this way, speechbrain toolkit is needed. In eval set, the result is 
SASV: 3.24% SV: 3.99% SPF: 1.64%.

