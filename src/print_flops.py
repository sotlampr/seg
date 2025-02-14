#!/usr/bin/env python
"""
Copyright (C) 2025  Sotiris Lamrpinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import argparse
import sys

import torch
from calflops import calculate_flops

import \
    m2f, sam, unet, segmentation_pytorch, torchvision_models, mb_sam

MODULES = [m2f, sam, unet, torchvision_models, segmentation_pytorch, mb_sam]


def all_models():
    for module in MODULES:
        for model in module.models.keys():
            yield (module, model)


def analyze_model(model, x, y):
    flops, macs, params = calculate_flops(
        model=model, args=[x], output_as_string=False,
        print_results=False
    )
    print(f"{flops/1024**4:.2f} TFLOPS")


if __name__ == "__main__":
    models = {f"{k.__name__}/{v}": (k, v) for k, v in all_models()}

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models)
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    parser.add_argument("-m", "--mixed-precision", action="store_true")
    parser.add_argument(
        "-s", "--shape", type=int, nargs=2, default=(1024, 1024)
    )
    args = parser.parse_args()

    module, model_name = models[args.model]
    model = module.new(model_name, pretrained=False, optimize=False)
    model_input = torch.randn(args.batch_size, 3, *args.shape)
    model_output = torch.randn(args.batch_size, 1, *args.shape)
    res = analyze_model(model, model_input, model_output)
    sys.exit(0)
