#!/usr/bin/env python3
"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
----------
Combine corrective annotations with original model segmentations
to create a synthetic densely annotated set.
"""
import argparse
import os

import numpy as np
from PIL import Image


def main(args):
    for fname in os.listdir(args.annotations_dir):
        seg_fname = os.path.join(args.segmentations_dir, fname)
        if not os.path.exists(seg_fname):
            print(f"WARNING: {fname} does not "
                  f"exist in {args.segmentations_dir}")
            continue
        ann_fname = os.path.join(args.annotations_dir, fname)
        with Image.open(ann_fname) as ann_im, Image.open(seg_fname) as seg_im:
            base_name, ext = os.path.splitext(fname)
            ann_arr, seg_arr = \
                np.array(ann_im)[:, :, :3], np.array(seg_im)[:, :, :3]

            seg_arr = (seg_arr > 0).any(2).astype(np.uint8)
            seg_arr[seg_arr > 0] = 255

            # red means addition
            seg_arr[ann_arr[:, :, 0] > 0] = 255

            # green means subtraction
            seg_arr[ann_arr[:, :, 1] > 0] = 0

            Image.fromarray(seg_arr).save(os.path.join(args.output_dir, fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_dir")
    parser.add_argument("segmentations_dir")
    parser.add_argument("output_dir")
    main(parser.parse_args())
