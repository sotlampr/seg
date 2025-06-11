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
    m2f, mb_sam, rootnav, sam, samII, segmentation_pytorch, segroot, \
    torchvision_models, unet, unet_valid  # noqa: F401 E401

MODULES = [
    m2f, mb_sam, rootnav, sam, samII, segmentation_pytorch, segroot,
    torchvision_models, unet, unet_valid
]


def all_models():
    for module in MODULES:
        for model in module.models.keys():
            yield (module, model)


def analyze_model(model, input_shape):
    flops, macs, params = calculate_flops(
        model=model, input_shape=input_shape,
        output_as_string=False,
        print_results=False
    )
    print(flops)


if __name__ == "__main__":
    models = {f"{k.__name__}/{v}": (k, v) for k, v in all_models()}

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models)
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    parser.add_argument(
        "-s", "--shape", type=int, nargs=2, default=(1024, 1024)
    )
    args = parser.parse_args()

    module, model_name = models[args.model]
    model = module.new(model_name, pretrained=False, optimize=False)
    input_shape = (args.batch_size, 3, *args.shape)
    res = analyze_model(model, input_shape)
    sys.exit(0)
