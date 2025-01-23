#!/bin/sh
python << END | xargs -n1 curl -LO --output-dir pretrained 
import sys
import urllib.request
sys.path.insert(0, 'src')
from seg import all_models
ALL_MODELS = list(all_models())
for module, model in ALL_MODELS:
    if hasattr(module, "get_url"):
      print(module.get_url(*module.models[model]))
END
