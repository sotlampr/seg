#!/bin/sh
# Copyright (C) 2025  Sotiris Lamrpinidis
# 
# This program is free software and all terms of the GNU General Public License
# version 3 as published by the Free Software Foundation apply. See the LICENSE
# file in the root directory of the project or <https://www.gnu.org/licenses/>
# for more details.
mkdir -p pretrained
python << END | xargs -n1 curl -C - -LO --output-dir pretrained 
import sys
import urllib.request
sys.path.insert(0, "Mask2Former")
sys.path.insert(0, "RootNav-2.0/training")
sys.path.insert(0, "SegRoot/code")
sys.path.insert(0, 'src')
from seg import all_models
ALL_MODELS = list(all_models())
for module, model in ALL_MODELS:
    if hasattr(module, "get_url"):
      print(module.get_url(*module.models[model]))
END
