export KALDI_ROOT=/Users/wuyaowang/kaldi/ # Your own kaldi path
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

SDT_ROOT=`pwd`
export PYTHONPATH=$SDT_ROOT:$PYTHONPATH
export PYTHONPATH=$SDT_ROOT/local:$PYTHONPATH
export PYTHONPATH=$SDT_ROOT/scripts:$PYTHONPATH

