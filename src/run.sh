#!/bin/bash
# Copyright (C) 2025, 2026  Sotiris Lamprinidis
# 
# This program is free software and all terms of the GNU General Public License
# version 3 as published by the Free Software Foundation apply. See the LICENSE
# file in the root directory of the project or <https://www.gnu.org/licenses/>
# for more details.
# ----------
# Script to run all experiments from the study [TODO: paper title].
# Example:
#   DATASETS="chicory cotton" LEARNING_RATE=1e-4 ./run,sh
# Or for specific models:
#   ./run,sh m2f/R50 m2f/swin-small
set -e

# These variables can be overrided in the calling environment
DATA_PATH=${DATA_PATH:=$HOME/src/seg/data}
if test ${HOSTNAME} = "triton"; then
  DATASETS="chicory grassland peanut sunflower"
elif test ${HOSTNAME} = "prime"; then
 DATASETS="cotton deeproot_corrective papaya sesame switchgrass"
else
  echo "Unkown host '$HOSTNAME'"
  exit 2
fi
NUM_RUNS=${NUM_RUNS:=2}
OUTDIR=${OUTDIR:=../results-ablation-patch-size}
NUM_WORKERS=${NUM_WORKERS:=20}
LEARNING_RATE=${LEARNING_RATE:=1e-4}

TODO="mb_sam/vit_t segmentation_pytorch/manet-inceptionv4"

die() {
    echo "Error on line $1"
    shift
    echo $@
}

trap 'die $LINENO' ERR

# samII/hiera-small-cotton-false
batch_size () {
  case $model in
    segmentation_pytorch/manet-inceptionv4)
      echo 8
      ;;
    mb_sam/vit_t)
      echo 16
      ;;
    *)
      echo "model $model not recognized"
      exit 2
      ;;
    esac
}

shape () {
  model=$1
  case $model in
    m2f*|mb_sam*|sam*|segmentation_pytorch/segformer*)
      echo 576 576;;
    segmentation_pytorch/*|rootnav*|segroot*)
      echo 1024 1024;;
    *)
  esac
}

shuffle() {
  echo $@ | tr ' ' '\n'| shuf | tr '\n' ' ';
}

# Use command-line arguments as todo if provided
if test $# -gt 0; then
  TODO="$@"
fi

# TODO=$(shuffle $TODO)
timeout=7200
echo TODO: $TODO
for i in $(seq 1 $NUM_RUNS); do
  for model in $TODO; do
    model_name=$(echo $model| tr / -)
    for dataset in $DATASETS; do
      for pretrain in false true; do
        out_dir="$OUTDIR/$model_name-$dataset-$i"

        # handle pretrained case - add appropriate flag
        pt_flag=""
        if test $pretrain = true; then
          pt_flag="-P"
          out_dir="$out_dir-pretrained"
        fi
        echo "$model $dataset pretrain=$pretrain"

        # skip if 'best' file exists, mean it's done
        if test -e $out_dir/best; then
          echo "	Already done, continue"
          continue
        fi

        bs=$(batch_size $model-$dataset-$pretrain)
        lr=$LEARNING_RATE
        if test $lr = "nan"; then continue; fi
        s=$(shape $model)
          
        # automatic mixed precision
        amp_arg=-m
        # gradient clipping
        clip_arg=-C
        workers_flag=-j$NUM_WORKERS
        extra_args="-w2 $pt_flag -a$lr -b$bs"
        extra_args="$extra_args -s $s -t$timeout $amp_arg $clip_arg"
        echo "	LR             = $lr"
        echo "	BATCH_SIZE     = $bs"
        echo "	SHAPE          = $s"
        echo "	ARGS           = $extra_args"
        extra_args="$extra_args"

        seg $model $dataset -o$out_dir $workers_flag $extra_args || echo $model-$dataset-$pretrain >> failed
      done
    done
  done
done
