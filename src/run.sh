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

# These variables can be overriden in the calling environment
DATA_PATH=${DATA_PATH:=../../seg_root_in_soil_next/data}
DATASETS=${DATASETS:=chicory cotton grassland papaya peanut sesame sunflower switchgrass deeproot_ann}
NUM_RUNS=${NUM_RUNS:=2}
OUTDIR=${OUTDIR:=../results}
NUM_WORKERS=${NUM_WORKERS:=20}
PRETRAINED=${PRETRAINED:=false true}
LEARNING_RATE=${LEARNING_RATE:=1e-3}

# The models to run can be overriden by passing positional arguments
TODO="\
  m2f/R50 \
  m2f/swin-small \
  m2f/swin-tiny \
  mb_sam/vit_t \
  samII/hiera-small \
  samII/hiera-base_plus \
  rootnav/hourglass \
  segmentation_pytorch/deeplabv3+-resnet50 \
  segmentation_pytorch/linknet-inceptionv4 \
  segmentation_pytorch/linknet-resnet50 \
  segmentation_pytorch/manet-inceptionv4 \
  segmentation_pytorch/manet-resnet50 \
  segmentation_pytorch/segformer-mit_b1 \
  segmentation_pytorch/segformer-mit_b2 \
  segmentation_pytorch/segformer-mit_b3 \
  segmentation_pytorch/unet++-inceptionv4 \
  segmentation_pytorch/unet++-resnet50 \
  segroot/w8d5 \
  torchvision_models/deeplabv3-resnet50 \
  unet_valid/GN \
  unet_valid/GNRes \
"

die() {
    echo "Error on line $1"
    shift
    echo $@
}

trap 'die $LINENO' ERR

# samII/hiera-small-cotton-false
batch_size () {
  run_id=$1
  case $run_id in
    segmentation_pytorch/segformer-mit_b1-chicory-true|\
    segmentation_pytorch/segformer-mit_b1-deeproot_ann-true|\
    segmentation_pytorch/segformer-mit_b1-grassland-true|\
    segmentation_pytorch/segformer-mit_b2-chicory-true|\
    segmentation_pytorch/segformer-mit_b2-deeproot_ann-true|\
    segmentation_pytorch/segformer-mit_b2-grassland-true|\
    segmentation_pytorch/segformer-mit_b3-chicory-*|\
    segmentation_pytorch/segformer-mit_b3-deeproot_ann-*|\
    segmentation_pytorch/segformer-mit_b3-grassland-*)
      echo 2;;
    m2f/R50-chicory-*|\
    m2f/R50-deeproot_ann-*|\
    m2f/R50-grassland-*|\
    m2f/swin-small-chicory-*|\
    m2f/swin-small-deeproot_ann-*|\
    m2f/swin-small-grassland-*|\
    m2f/swin-tiny-chicory-*|\
    m2f/swin-tiny-deeproot_ann-*|\
    m2f/swin-tiny-grassland-*|\
    rootnav/hourglass-chicory-*|\
    samII/hiera-base_plus-chicory-*|\
    samII/hiera-base_plus-cotton-*|\
    samII/hiera-base_plus-deeproot_ann-*|\
    samII/hiera-base_plus-grassland-*|\
    samII/hiera-base_plus-papaya-*|\
    samII/hiera-base_plus-peanut-*|\
    samII/hiera-base_plus-sesame-*|\
    samII/hiera-base_plus-sunflower-*|\
    samII/hiera-base_plus-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b1-chicory-false|\
    segmentation_pytorch/segformer-mit_b1-deeproot_ann-false|\
    segmentation_pytorch/segformer-mit_b1-grassland-false|\
    segmentation_pytorch/segformer-mit_b2-chicory-false|\
    segmentation_pytorch/segformer-mit_b2-deeproot_ann-false|\
    segmentation_pytorch/segformer-mit_b2-grassland-false)
      echo 4;;
    m2f/swin-small-cotton-*|\
    m2f/swin-small-papaya-*|\
    m2f/swin-small-peanut-*|\
    m2f/swin-small-sesame-*|\
    m2f/swin-small-sunflower-*|\
    m2f/swin-small-switchgrass-*|\
    m2f/swin-tiny-cotton-*|\
    m2f/swin-tiny-papaya-*|\
    m2f/swin-tiny-peanut-*|\
    m2f/swin-tiny-sesame-*|\
    m2f/swin-tiny-switchgrass-*|\
    mb_sam/vit_t-cotton-*|\
    mb_sam/vit_t-grassland-false|\
    rootnav/hourglass-cotton-*|\
    rootnav/hourglass-deeproot_ann-*|\
    rootnav/hourglass-grassland-*|\
    rootnav/hourglass-papaya-*|\
    rootnav/hourglass-peanut-*|\
    rootnav/hourglass-sesame-*|\
    rootnav/hourglass-sunflower-*|\
    rootnav/hourglass-switchgrass-*|\
    samII/hiera-base_plus-sesame-false|\
    samII/hiera-small-chicory-*|\
    samII/hiera-small-cotton-*|\
    samII/hiera-small-deeproot_ann-*|\
    samII/hiera-small-grassland-*|\
    samII/hiera-small-papaya-*|\
    samII/hiera-small-peanut-*|\
    samII/hiera-small-sesame-*|\
    samII/hiera-small-sunflower-*|\
    samII/hiera-small-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b1-cotton-*|\
    segmentation_pytorch/segformer-mit_b1-papaya-*|\
    segmentation_pytorch/segformer-mit_b1-peanut-true|\
    segmentation_pytorch/segformer-mit_b1-sesame-*|\
    segmentation_pytorch/segformer-mit_b1-sunflower-true|\
    segmentation_pytorch/segformer-mit_b1-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b2-cotton-*|\
    segmentation_pytorch/segformer-mit_b2-papaya-*|\
    segmentation_pytorch/segformer-mit_b2-peanut-*|\
    segmentation_pytorch/segformer-mit_b2-sesame-*|\
    segmentation_pytorch/segformer-mit_b2-sunflower-true|\
    segmentation_pytorch/segformer-mit_b2-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b3-cotton-*|\
    segmentation_pytorch/segformer-mit_b3-papaya-*|\
    segmentation_pytorch/segformer-mit_b3-peanut-*|\
    segmentation_pytorch/segformer-mit_b3-sesame-*|\
    segmentation_pytorch/segformer-mit_b3-sunflower-*|\
    segmentation_pytorch/segformer-mit_b3-switchgrass-*)
      echo 8;;
    *)
      echo 16;;
    esac
}

shape () {
  model=$1
  case $model in
    m2f*|mb_sam*|sam*|segmentation_pytorch/segformer*)
      echo 1024 1024;;
    segmentation_pytorch/*|rootnav*|segroot*)
      echo 576 576;;
    *)
      echo 572 572;;
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
      for pretrain in $PRETRAINED; do
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
        case $dataset in
          *_ann)
            # sparse annotations
            echo "	SPARSE ANNOTATIONS"
            extra_args="$extra_args -s $s -t$timeout $amp_arg $clip_arg -A";;
          *)
            extra_args="$extra_args -s $s -t$timeout $amp_arg $clip_arg";;
        esac;
        echo "	LR             = $lr"
        echo "	BATCH_SIZE     = $bs"
        echo "	SHAPE          = $s"
        echo "	ARGS           = $extra_args"
        extra_args="$extra_args"

        ./seg.py $DATA_PATH/$dataset $model -o$out_dir $workers_flag $extra_args || echo $model-$dataset-$pretrain >> failed
      done
    done
  done
done
