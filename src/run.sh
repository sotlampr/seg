#!/bin/bash
set -e

# These variables can be overrided in the calling environment
DATA_PATH=${DATA_PATH:=../../seg_root_in_soil_next/data}
DATASETS=${DATASETS:=chicory cotton grassland papaya peanut sesame sunflower switchgrass deeproot_ann}
NUM_RUNS=${NUM_RUNS:=2}
OUTDIR=${OUTDIR:=../results}
NUM_WORKERS=${NUM_WORKERS:=20}

# The models to run can be overrided by passing positional arguments
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
  segroot/w16d4 \
  segroot/w32d5 \
  segroot/w64d4 \
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

batch_size () {
  run_id=$1
  case $run_id in
    *)
      echo 16;;
    esac
}

learning_rate () {
  run_id=$1
  case $run_id in
    *)
      echo 1e-3;;
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
        lr=$(learning_rate $model-$dataset-$pretrain)
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
