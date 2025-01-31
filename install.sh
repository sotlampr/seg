#!/bin/bash
# Copyright (C) 2025  Sotiris Lamrpinidis
# 
# This program is free software and all terms of the GNU General Public License
# version 3 as published by the Free Software Foundation apply. See the LICENSE
# file in the root directory of the project or <https://www.gnu.org/licenses/>
# for more details.
echo "$(tput bold)===> Installing python requirements with pip$(tput sgr0)"
pip install -r requirements.txt

echo "$(tput bold)===> Installing detectron2$(tput sgr0)"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' \
  --no-build-isolation

echo "$(tput bold)===> Installing Mask2Former$(tput sgr0)"
git clone 'https://github.com/facebookresearch/Mask2Former.git'
pip install -r Mask2Former/requirements.txt
echo "$(tput bold)===> Compiling mask2former kernels$(tput sgr0)"
pushd Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh
popd
