import pickle as pk
import kaldi_io
import numpy as np
import argparse
import os

def main(args):
    pre_embd_path = args.pre_embd_path
    embd_file = args.embd_file
    ark_file = args.out_ark_file
    scp_file = args.out_scp_file

    with open(embd_file, "rb") as f:
        emb = pk.load(f)

    print(len(emb))

    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:' + ark_file + "," + scp_file
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    # with open(ark_file,'wb') as f:
        for key in sorted(emb.keys()):
          vec = emb[key].reshape(1,-1)
          kaldi_io.write_mat(f, vec, key=key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Make ASVspoof embeddings")
    parser.add_argument("--pre_embd_path", required=True, help="The init pre_emb path")
    parser.add_argument("--embd_file", required=True, help="The init embd file")
    parser.add_argument("--out_ark_file", required=True, help="The init ark file path")
    parser.add_argument("--out_scp_file", required=True, help="The init scp file path")
    args = parser.parse_args()
    main(args)