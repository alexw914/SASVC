import pickle as pk
import kaldi_io
from kaldiio import WriteHelper
import numpy as np
import argparse
import os

def main(args):

    embd_file = args.embd_file
    ark_file = args.out_ark_file
    scp_file = args.out_scp_file

    with open(embd_file, "rb") as f:
        emb = pk.load(f)

    print("This {} file has {} trails.".format(embd_file, len(emb)))

    # with open(ark_file,'wb') as f:
    #     for key in sorted(emb.keys()):
    #       vec = emb[key]
    #       kaldi_io.write_vec_flt(f, vec, key=key)

    ark_scp_output = 'ark,scp:'+ark_file+','+scp_file
    with WriteHelper(ark_scp_output) as writer:
        for key in emb.keys():
          vec = emb[key]
          writer(key,vec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Make ASVspoof embeddings")
    parser.add_argument("--embd_file", required=True, help="The init embd file")
    parser.add_argument("--out_ark_file", required=True, help="The init ark file path")
    parser.add_argument("--out_scp_file", required=True, help="The init scp file path")
    args = parser.parse_args()
    main(args)