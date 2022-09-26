import argparse
import os
from pathlib import Path

def main(args):
    init_protocol_path = args.init_protocol_path
    init_wav_path = args.init_wav_path
    init_save_path = args.init_save_path
    Path(init_save_path).mkdir(exist_ok=True)
    with open(init_protocol_path, "r") as f:
        lines = f.readlines()
    lines = sorted(lines)
    with open(os.path.join(init_save_path,"wav.scp"),"w") as fw:
        with open(os.path.join(init_save_path,"utt2spk"),"w") as fw2: 
            for line in lines:
                spk,utt,__,att,bonafife = line.split(" ")
                file_name = utt+".flac"
                fw.write(f"{utt} {os.path.join(init_wav_path, file_name)}\n")
                fw2.write(f"{utt} {spk}\n")

    print("Prepare init scp of {} finished!".format(os.path.join(init_save_path,"wav.scp")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Make ASVspoof")
    parser.add_argument("--init_protocol_path", required=True, help="The init protocol path")
    parser.add_argument("--init_wav_path", required=True, help="The init wav path")
    parser.add_argument("--init_save_path", required=True, help="The init utt2spk path")
    args = parser.parse_args()
    main(args)
