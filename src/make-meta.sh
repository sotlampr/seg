#!/bin/bash
# Create a ./meta file on every directory in the argument list
# with model size, num. parameters, and flops.

DATA_PATH=${DATA_PATH:=../../seg_root_in_soil_next/data}

for model_path in $@; do
  meta_fn=$model_path/meta
  if test -e $meta_fn; then
    echo $model_path: meta found, continue
    continue
  fi

  if echo $f | grep -q -- -pretrained; then
    pt_flag=-P
  fi

  tmp=$(echo $model_path| cut -f3 -d/ | sed 's/-pretrained//g')
  dataset=$(echo $tmp| rev| cut -f2 -d-|rev| tr '-' '	')
  read package model <<< $(echo $tmp| rev| cut -f3- -d-|rev| sed 's/-/	/')
  
  cfg_fn=$model_path/config
  if ! test -e $cfg_fn; then
    echo "$model_path/config does not exist, skipping"
    continue
  fi

  bs=$(grep batch_size $cfg_fn| cut -f2 -d'	')

  # Get model shape and image shape/resolution
  read height width <<< $(grep shape $cfg_fn| grep -o '[0-9]\+'| tr '\n' ' ')
  read img_height img_width <<< $(identify -format "%h %w" "$(ls -1 $DATA_PATH/$dataset/val/photos/*| tail -1)")

  if ! echo $model_path| grep -Eq '(rootnav|segroot)'; then
    # Use the smallest of the two
    if test $img_height -lt $height; then height=$img_height; fi
    if test $img_width -lt $width; then width=$img_width; fi
  fi


  if echo $model_path| grep -q segmentation_pytorch; then
    # These models need input divisible by 16/32
    if test $(($height%32)) -ne 0; then
      height=$(($height+32-$height%32))
    fi
    if test $(($width%32)) -ne 0; then
      width=$(($width+32-$width%32))
    fi
  fi

  shape="$height $width"

  fsize=$(du $model_path/checkpoint_best.pth| cut -f1)
  if test $? -ne 0; then echo "$model_path: FAILED"; continue; fi

  num_params=$(python -c "import torch; x = torch.load('$model_path/checkpoint_best.pth'); print(sum(p.numel() for p in x.values()))")
  if test $? -ne 0; then echo "$model_path: FAILED"; continue; fi

  cached=$(grep "$model-$bs-$shape$pt_flag:" .meta-cache)
  if test $? -eq 0; then
    echo "$model_path: Found cached"
    flops=$(echo $cached| cut -f2 -d:)
  else
    flops=$(./print_flops.py $package/$model -b$bs -s $shape $pt_flag)
    if test $? -ne 0; then echo "$model_path: FAILED"; continue; fi
    echo "$model-$bs-$shape$pt_flag:$flops" >> .meta-cache
  fi

  echo "file_size	$fsize" > $meta_fn
  echo "num_params	$num_params" >> $meta_fn
  echo "flops	$flops" >> $meta_fn

done
