#!/usr/bin/env python3
"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
----------
Create `f1_score.csv` files in inferred folders. The file contains the score
for each image, as well as the micro average.
"""
import csv
import glob
import os
import sys

import torch
from torchvision.io import read_image as read_image_, ImageReadMode

from seg import f1_score_, tp_fp_fn
from seg_common import DATA_ROOT


def read_image(fn, mode="GRAY"):
    img = read_image_(fn, getattr(ImageReadMode, mode))
    return img.cuda()


def main(pred_folder, dataset, subset):
    sparse_annotations = dataset.endswith("_corrective")
    true_folder = os.path.join(DATA_ROOT, dataset, subset, "annotations")
    target_imgs = {
        os.path.basename(fn):
            read_image(fn, "RGB" if sparse_annotations else "GRAY")
        for fn in glob.glob(f"{true_folder}/*.png")
    }

    model_imgs = {fn: read_image(f"{pred_folder}/{fn}") for fn in target_imgs}

    if sparse_annotations:
        tps_fps_fns = {}
        for fn in target_imgs:
            # either red or green
            mask = target_imgs[fn][:2].any(0).bool()
            # red: positive class
            targ = target_imgs[fn][0]
            tps_fps_fns[fn] = tp_fp_fn(
                model_imgs[fn][0][mask] > 0, targ[mask] > 0)
    else:
        # tps fps fns expects logits, so we make sure 0s are negative
        tps_fps_fns = {
            fn: tp_fp_fn(model_imgs[fn] > 0, target_imgs[fn])
            for fn in target_imgs
        }

    tps, fps, fns = torch.tensor(list(tps_fps_fns.values())).sum(0)

    losses = {
        fn: f1_score_(*args).item()
        for fn, args in tps_fps_fns.items()
    }
    losses["micro"] = f1_score_(tps, fps, fns).item()

    with open(f"{pred_folder}/f1_score.csv", "w", newline=None) as fp:
        writer = csv.writer(fp)
        writer.writerow(("filename", "f1_score"))
        for k, v in losses.items():
            writer.writerow((k, v))

    return 0 


def cli_main():
    if len(sys.argv) != 4 or sys.argv[1] in {"-h", "--help"}:
        print(
            f"Usage: {sys.argv[0]} PREDICTIONS DATASET SUBSET\n"
            "where PREDICTIONS is a folder containing *.png files.")
        return 1
    return main(*sys.argv[1:])


if __name__ == "__main__":
    sys.exit(cli_main())
