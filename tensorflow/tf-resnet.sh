#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd $DIR/benchmarks/scripts/tf_cnn_benchmarks
/usr/bin/time -v python3 tf_cnn_benchmarks.py --forward_only=True --data_format=NHWC --device=cpu --batch_size=64 --num_inter_threads=0 --num_intra_threads=2 --nodistortions --model=resnet50 --kmp_blocktime=0 --num_batches=20 --num_warmup_batches 0
