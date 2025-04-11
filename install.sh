#!/bin/bash
# Copyright (C) 2025  Sotiris Lamrpinidis
# 
# This program is free software and all terms of the GNU General Public License
# version 3 as published by the Free Software Foundation apply. See the LICENSE
# file in the root directory of the project or <https://www.gnu.org/licenses/>
# for more details.
echo "$(tput bold)===> Installing detectron2$(tput sgr0)"
if pip freeze | grep -q detectron2 >/dev/null; then
  echo "$(tput bold)=====> Already installed$(tput sgr0)"
else
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' \
    --no-build-isolation
fi

echo "$(tput bold)===> Installing Mask2Former$(tput sgr0)"
if test -d ./Mask2Former; then
  echo "$(tput bold)=====> Already installed$(tput sgr0)"
else
  git clone 'https://github.com/facebookresearch/Mask2Former.git'
  pip install -r Mask2Former/requirements.txt
  echo "$(tput bold)===> Compiling mask2former kernels$(tput sgr0)"
  pushd Mask2Former/mask2former/modeling/pixel_decoder/ops
  sh make.sh
  popd
fi


echo "$(tput bold)===> Installing RootNav$(tput sgr0)"
if test -d ./RootNav-2.0; then
  echo "$(tput bold)=====> Already installed$(tput sgr0)"
else
  git clone https://github.com/robail-yasrab/RootNav-2.0/
  pip install -r RootNav-2.0/inference/requirements.txt
fi

echo "$(tput bold)===> Installing SegRoot$(tput sgr0)"
if test -d ./SegRoot; then
  echo "$(tput bold)=====> Already installed$(tput sgr0)"
else
  git clone https://github.com/wtwtwt0330/SegRoot
  pip install -r RootNav-2.0/inference/requirements.txt
fi
