#!/bin/bash
set -e

# These variables can be overrided in the calling environment
DATA_PATH=${DATA_PATH:=../data}
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
    segmentation_pytorch/segformer-mit_b1-deeproot_ann-true|\
    segmentation_pytorch/segformer-mit_b2-deeproot_ann-true|\
    segmentation_pytorch/segformer-mit_b3-deeproot_ann-*)
      echo 2;;
    m2f/R50-deeproot_ann-*|\
    m2f/swin-small-deeproot_ann-*|\
    m2f/swin-tiny-deeproot_ann-*|\
    samII/hiera-base_plus-cotton-*|\
    samII/hiera-base_plus-papaya-*|\
    segmentation_pytorch/segformer-mit_b1-deeproot_ann-false|\
    segmentation_pytorch/segformer-mit_b2-deeproot_ann-false|\
    samII/hiera-base_plus-sesame-*|\
    samII/hiera-base_plus-deeproot_ann-*|\
    samII/hiera-base_plus-switchgrass-*)
      echo 4;;
    m2f/swin-small-cotton-*|\
    m2f/swin-small-papaya-*|\
    m2f/swin-small-sesame-*|\
    m2f/swin-small-switchgrass-*|\
    m2f/swin-tiny-cotton-*|\
    m2f/swin-tiny-papaya-*|\
    m2f/swin-tiny-sesame-*|\
    m2f/swin-tiny-switchgrass-*|\
    mb_sam/vit_t-cotton-*|\
    rootnav/hourglass-cotton-*|\
    rootnav/hourglass-deeproot_ann-*|\
    rootnav/hourglass-papaya-*|\
    rootnav/hourglass-sesame-*|\
    rootnav/hourglass-switchgrass-*|\
    samII/hiera-base_plus-sesame-false|\
    samII/hiera-small-cotton-*|\
    samII/hiera-small-deeproot_ann-*|\
    samII/hiera-small-papaya-*|\
    samII/hiera-small-sesame-*|\
    samII/hiera-small-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b1-cotton-*|\
    segmentation_pytorch/segformer-mit_b1-papaya-*|\
    segmentation_pytorch/segformer-mit_b1-sesame-*|\
    segmentation_pytorch/segformer-mit_b1-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b2-cotton-*|\
    segmentation_pytorch/segformer-mit_b2-papaya-*|\
    segmentation_pytorch/segformer-mit_b2-sesame-*|\
    segmentation_pytorch/segformer-mit_b2-switchgrass-*|\
    segmentation_pytorch/segformer-mit_b3-cotton-*|\
    segmentation_pytorch/segformer-mit_b3-papaya-*|\
    segmentation_pytorch/segformer-mit_b3-sesame-*|\
    segmentation_pytorch/segformer-mit_b3-switchgrass-*)
      echo 8;;
    *)
      echo 16;;
    esac
}

learning_rate () {
  run_id=$1
  case $run_id in
    m2f/R50-deeproot_ann-*|\
    m2f/swin-small-deeproot_ann-*|\
    m2f/swin-tiny-deeproot_ann-*|\
    samII/hiera-base_plus-cotton-false|\
    samII/hiera-base_plus-deeproot_ann-*|\
    samII/hiera-base_plus-sesame-false|\
    samII/hiera-small-cotton-false|\
    samII/hiera-small-deeproot_ann-*|\
    samII/hiera-small-sesame-false|\
    segmentation_pytorch/manet-inceptionv4-deeproot_ann-false|\
    segmentation_pytorch/manet-resnet50-deeproot_ann-false|\
    segroot/w8d5-deeproot_ann-false|\
    unet_valid/GN-deeproot_ann-false|\
    unet_valid/GNRes-deeproot_ann-false)
      echo "nan";;
    mb_sam/vit_t-cotton-false|\
    segroot-w64d4-cotton-*)
      echo 1e-2;;
    m2f/R50-deeproot_ann-true|\
    m2f/swin-small-deeproot_ann-true|\
    m2f/swin-tiny-deeproot_ann-true|\
    mb_sam/vit_t-deeproot_ann-false|\
    rootnav/hourglass-deeproot_ann-false|\
    samII/hiera-base_plus-deeproot_ann-true|\
    samII/hiera-base_plus-switchgrass-false|\
    samII/hiera-small-deeproot_ann-true|\
    segmentation_pytorch/deeplabv3+-resnet50-deeproot_ann-false|\
    segmentation_pytorch/linknet-inceptionv4-deeproot_ann-false|\
    segmentation_pytorch/manet-resnet50-deeproot_ann-true|\
    segmentation_pytorch/segformer-mit_b1-deeproot_ann-*|\
    segmentation_pytorch/segformer-mit_b2-deeproot_ann-*|\
    segmentation_pytorch/segformer-mit_b3-deeproot_ann-*|\
    segmentation_pytorch/unet++-inceptionv4-deeproot_ann-false|\
    segmentation_pytorch/unet++-resnet50-deeproot_ann-false|\
    segroot/w16d4-deeproot_ann-false|\
    segroot/w32d5-deeproot_ann-false|\
    segroot/w64d4-deeproot_ann-false|\
    torchvision_models/deeplabv3-resnet50-deeproot_ann-false)
      echo 1e-4;;
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
