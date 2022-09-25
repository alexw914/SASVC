# SASVC
Spoofing-Aware Speaker Verification

This project provide three ways to realize Sooofing-aware Speaker Verification system. Score Fusion, embedding fusion and muti-task learning

## Score Fusion
It takes PLDA backend as classifier, in this way. Unsupervised domain adaptation and supervised domain adaptation were applied in score fusion method to improve the speaker verification performance. It needs kaldi and asv-subtools.

## Emd Fusion
Using Conv1D layer and SEModule to train a SASV model using embeddings from pre-trained asv system and countermeature system. 

## Multi-task
In this method, i used a pretrained asv system and Attentive statistic pooling layers and backend to build a SASV model. In this way, speechbrain toolkit is needed

It's not finished!!!
