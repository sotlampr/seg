#!/usr/bin/env python
"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
"""
import argparse
import sys

from calflops import calculate_flops

from seg_common import all_models


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
    parser.add_argument("-P", "--pretrained", action="store_true")
    args = parser.parse_args()

    module, model_name = models[args.model]
    model = module.new(model_name, pretrained=args.pretrained, optimize=False)
    input_shape = (args.batch_size, 3, *args.shape)
    res = analyze_model(model, input_shape)
    sys.exit(0)
