#!/bin/sh
# Output tsv-like values to stdout from ../results
# Accepts and optional argument "results" to generate fine-grained results for each
# epoch instead of just the best model

if test $# -eq 1; then
  target=$1
else
  target=best
fi

echo "package model pretrained dataset run epoch step training_time train_loss val_loss val_iou val_f1 batch_size learning_rate file_size num_params flops"| tr ' ' '	'
for f in ../results/**/$target; do
  if echo $f | grep -q -- -pretrained; then
    pt=1
  else
    pt=0
  fi
  fn=$(echo $f| cut -f3 -d/ | sed 's/-pretrained//g')
  dataset_run=$(echo $fn| rev| cut -f1,2 -d-|rev| tr '-' '	')
  package_model=$(echo $fn| rev| cut -f3- -d-|rev| sed 's/-/	/')
  cfg_fn=$(echo $f| sed "s/$target$/config/g")
  if ! test -e $cfg_fn; then
    echo ERR: $cfg_fn not found 1>&2
    continue;
  fi

  bs=$(grep batch_size $cfg_fn| cut -f2 -d'	')
  lr=$(grep learning_rate $cfg_fn| cut -f2 -d'	')

  meta_fn=$(echo $f| sed "s/$target$/meta/g")
  if ! test -e $meta_fn; then
    echo ERR: $meta_fn not found 1>&2
    continue;
  fi
  file_size=$(grep file_size $meta_fn| cut -f2 -d'	')
  num_params=$(grep num_params $meta_fn| cut -f2 -d'	')
  flops=$(grep flops $meta_fn| cut -f2 -d'	')

  if test $target = "best"; then
    echo $package_model $pt $dataset_run $(cat $f) $bs $lr $file_size $num_params $flops| tr  ' ' '	'
  elif test $target = "results"; then
    echo "$(sed 1d $f)"| sed -e "s/^/$package_model $pt $dataset_run /g" -e "s/$/ $bs $lr $file_size $num_params $flops/g" | tr  ' ' '	'
  else
    echo "Unrecognized target '$target'"
    exit 1
  fi
done
