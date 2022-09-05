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
trials = open(sys.argv[1], 'r').readlines()
scores = open(sys.argv[2], 'r').readlines()
spkrutt2target = {}
for line in trials:
  spkr, utt, att, target = line.strip().split()
  spkrutt2target[spkr+utt]=target

with open("./asv_score.txt","w") as f:
  for line in scores:
    spkr, utt, score = line.strip().split()
    f.write(f"{score} {spkrutt2target[spkr+utt]}\n")
