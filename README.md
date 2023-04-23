# SASVC
Spoofing-Aware Speaker Verification
More information can be found in the [webpage](https://sasv-challenge.github.io).
This project provide three ways to realize Sooofing-aware Speaker Verification system. Score Fusion, embedding fusion and muti-task learning.

# Prepare

##### 1. Prepare embedding file
    You can refer this offical [project](https://github.com/search?q=SASVC2022). Download the Embedding file. Put the embeddings floder to SF and EF folder.

##### 2. Tools
    This project need install this tools first
    1. [kaldi](https://github.com/kaldi-asr/kaldi)
    2. [ASV-subtools](https://github.com/Snowdar/asv-subtools)
    3. [speechbrain](https://github.com/search?q=speechbrain)


## Score Fusion
$$S_{sasv} = S_{cm} * sigmoid（S_{asv}）$$
##### ASV: ECAPA-TDNN + PLDA + LIP-Reg Adaptation SV-EER 1.47%
It takes PLDA backend as classifier, in this way. Unsupervised domain adaptation and supervised domain adaptation were applied in score fusion method to improve the speaker verification performance.  
##### CM: wav2vev-ASSIST(Pretrain + Finetune CM-EER 0.20%)
 score produced by Wav2Vec-AASIST EER was 0.20%. The sasv score is the multiplication of asv score and cm score processed by sigmoid function.

```
$ cd SF && ./run.sh
```


## Embedding Fusion

Using Conv1D layers and SEModule to train a SASV model using embeddings from pre-trained asv system and countermeature system provided by SASV baseline system.

Training it:
```
cd EF && python main.py --config ./configs/sasv.conf
```

## Multi-task
In this method, i used a pretrained asv system and Attentive statistic pooling layers and fusion backend to build a SASV model. In this way, speechbrain toolkit is needed.

Train:
```
cd EF && python train.py yaml/sasv.yaml
```
Test:
```
python eval.py yaml/sasv.yaml
```


## Results(Eval set):

| Model | SASV(EER%) | SV(EER%) | SPF(EER%) |
|:------|:------------:|:------------:|:------------:|
| Score Fusion | 1.06 | 1.53 | **0.64** |
| Embedding Fusion | **0.96** | **1.24** | 0.68 |
| Multi-task | 3.24 | 3.99 | 1.64 | 
