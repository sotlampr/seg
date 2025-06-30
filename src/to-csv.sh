#!/bin/sh

if test $# -eq 1; then
  target=$1
else
  target=best
fi
#
#echo "is_valid is_1024 scale_loss no_empty our_augment run epoch step training_time train_loss val_loss val_iou val_f1"| tr ' ' ' '
#for f in  ../ablation/**/$target; do
#  fn=$(echo $f| cut -f1 -d' ' |sed -e 's/\/'"$target"'//' -e 's/.*\///')
#  prefix="$(echo $fn| cut -d- -f1| sed -e 's/unet_nopad_GN/1/g' -e 's/unet_GN/0/g')      $(echo $fn| cut -d- -f2-| tr '-' '      '| sed -e 's/X/0/g' -e 's/[a-z_]\+/1/g' -e 's/1024/1/g' -e 's/576/0/g')"
#  if test $target = "best"; then
#    echo "$prefix	$(cat $f)"
#  else
#    echo "$(sed -e '1,1d' -e 's/^/'"$prefix"'/g' $f)"
#  fi
#done


#echo "package model pretrained dataset run epoch step training_time train_loss val_loss val_iou val_f1 batch_size model_size num_parms flops"| tr ' ' '	'
echo "package model pretrained dataset run epoch step training_time train_loss val_loss val_iou val_f1"| tr ' ' '	'
for f in ../results/**/$target; do
  if echo $f | grep -q -- -pretrained; then
    pt=1
  else
    pt=0
  fi
  fn=$(echo $f| cut -f3 -d/ | sed 's/-pretrained//g')
  dataset_run=$(echo $fn| rev| cut -f1,2 -d-|rev| tr '-' '	')
  package_model=$(echo $fn| rev| cut -f3- -d-|rev| sed 's/-/	/')
  meta_fn=$(echo $f| sed "s/$target$/meta/g")

  if test $target = "best"; then
    echo $package_model $pt $dataset_run $(cat $f) | tr  ' ' '	'
  elif test $target = "results"; then
    echo "$(sed 1d $f)"| sed -e "s/^/$package_model $pt $dataset_run /g" | tr  ' ' '	'
  else
    echo "Unrecognized target '$target'"
    exit 1
  fi
done
