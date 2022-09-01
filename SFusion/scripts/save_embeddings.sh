set -e
pre_embd_path=$1
system=$2
save_path=$3
phase=$4

python scripts/save_embeddings.py --pre_embd_path $pre_embd_path \
		                          --embd_file $pre_embd_path/asv_embd_$phase.pk \
		                          --out_ark_file $save_path/embeddings/embd_$system.$phase.ark \
		                          --out_scp_file $save_path/embeddings/embd_$system.$phase.scp \
		       