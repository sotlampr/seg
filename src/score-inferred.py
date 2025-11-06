#!/usr/bin/env python3
"""Create `f1_score.csv` files in inferred folders."""
import csv
import glob
import os
import sys

import torch
from torchvision.io import read_image as read_image_, ImageReadMode

from seg import f1_score_, tp_fp_fn


def read_image(fn):
    img = read_image_(fn, ImageReadMode.GRAY)
    return img.cuda()


if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
    print(
        f"Usage: {sys.argv[0]} BASE_PATH\n"
        "where BASE_PATH contains annotations/*.png and any model folders.")
    sys.exit(1)

base_path = sys.argv[1]
target_imgs = {
    os.path.basename(fn): read_image(fn)
    for fn in glob.glob(f"{base_path}/annotations/*.png")
}


for model_path in glob.glob(f"{base_path}/*"):
    if model_path.endswith("annotations") or model_path.endswith("photos"):
        continue

    # tps fps fns expects logits, so we make sure 0s are negative
    model_imgs = {
        fn: read_image(f"{model_path}/{fn}") - 0.5
        for fn in target_imgs
    }

    tps_fps_fns = {
        fn: tp_fp_fn(model_imgs[fn], target_imgs[fn])
        for fn in target_imgs
    }

    tps, fps, fns = torch.tensor(list(tps_fps_fns.values())).sum(0)

    losses = {
        fn: f1_score_(*args).item()
        for fn, args in tps_fps_fns.items()
    }
    losses["micro"] = f1_score_(tps, fps, fns).item()

    with open(f"{model_path}/f1_score.csv", "w", newline=None) as fp:
        writer = csv.writer(fp)
        writer.writerow(("filename", "f1_score"))
        for k, v in losses.items():
            writer.writerow((k, v))
