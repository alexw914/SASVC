#!/usr/bin/env python3
#
# Copyright 2015   David Snyder
# Apache 2.0.
#
# Copied from egs/sre10/v1/local/prepare_for_eer.py (commit 9cb4c4c2fb0223ee90c38d98af11305074eb7ef8)
#
# Given a trials and scores file, this script
# prepares input for the binary compute-eer.
import sys
import numpy as np
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

trials = open(sys.argv[1], 'r').readlines()
asv_scores = open(sys.argv[2], 'r').readlines()
cm_scores = open(sys.argv[3], 'r').readlines()

spkrutt2target = {}

cm2score={}
for line in cm_scores:
  utt, att, bona, score = line.strip().split()
  cm2score[utt] = float(score)

for line in trials:
  spkr, utt, att, target = line.strip().split()
  spkrutt2target[spkr+utt]=target


with open(sys.argv[4],"w") as f:
  for line in asv_scores:
    spkr, utt, asv_score = line.strip().split()
    sasv_score = sigmoid(cm2score[utt])*sigmoid(float(asv_score))
    sasv_score = asv_score     
    f.write("{} {}\n".format(sasv_score, spkrutt2target[spkr+utt]))
