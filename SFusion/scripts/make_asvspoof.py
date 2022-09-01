import argparse
import os


def main(args):
    init_protocol_path = args.init_protocol_path
    init_wav_scp_file = args.init_wav_scp_file
    init_wav_path = args.init_wav_path
    init_utt2spk_path = args.init_utt2spk_path

    with open(init_protocol_path, "r") as f:
        lines = f.readlines()
    lines = sorted(lines)
    with open(init_wav_scp_file,"w") as fw:
        with open(init_utt2spk_path,"w") as fw2: 
            for line in lines:
                spk,utt,__,att,bonafife = line.split(" ")
                print(utt)
                file_name = utt+".flac"
                fw.write(f"{utt} {os.path.join(init_wav_path, file_name)}\n")
                fw2.write(f"{utt} {spk}\n")

    print("Prepare init wav.scp finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Make ASVspoof")
    parser.add_argument("--init_protocol_path", required=True, help="The init protocol path")
    parser.add_argument("--init_wav_scp_file", required=True, help="The init wav.scp file")
    parser.add_argument("--init_wav_path", required=True, help="The init wav path")
    parser.add_argument("--init_utt2spk_path", required=True, help="The init utt2spk path")
    args = parser.parse_args()
    main(args)
