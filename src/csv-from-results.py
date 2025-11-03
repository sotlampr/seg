#!/usr/bin/env python3
"""Generate a csv from training outputs."""
import argparse
import csv
import re
import os.path
import sys


directory_regexp = r"(?P<model>\S+)-(?P<dataset>\S+?)-(?P<run>[0-9])(?P<pretrained>-pretrained)?$"  # noqa: E501

out_keys = {
    "model", "dataset", "run", "pretrained", "file_size", "num_params",
    "flops", "batch_size", "eval_frequency", "learning_rate",
    "num_train", "num_val", "patience", "shape", "warmup_steps",
    "epoch", "step", "wall_duration", "train_loss", "val_loss", "val_iou",
    "val_fscore"
}


parser = argparse.ArgumentParser(
    description="Generate a csv with results from training outputs.")
parser.add_argument("checkpoint", nargs="+", help="checkpoint_best.pth paths")
args = parser.parse_args()

writer = csv.DictWriter(sys.stdout, fieldnames=sorted(list(out_keys)))
writer.writeheader()

for fn in args.checkpoint:
    path = os.path.dirname(fn)
    parent_dir = os.path.basename(path)
    meta = re.match(directory_regexp, parent_dir).groupdict()
    meta["pretrained"] = bool(meta["pretrained"])

    meta_fn = os.path.join(path, "meta")
    if not os.path.exists(meta_fn):
        print(f"WARNING: meta does not exist in {path}", file=sys.stderr)
    else:
        with open(meta_fn) as fp:
            for line in fp.readlines():
                k, v = str.split(str.strip(line))
                if k not in out_keys:
                    continue
                meta[k] = int(v)

    config_fn = os.path.join(path, "config")
    if not os.path.exists(config_fn):
        print(f"WARNING: config does not exist in {path}", file=sys.stderr)
    else:
        with open(config_fn) as fp:
            for line in fp.readlines():
                k, v = str.split(str.strip(line), maxsplit=1)
                if k not in out_keys:
                    continue
                if (m := re.match(r"\[(\d+), ?(\d+)\]", v)):
                    meta[k] = list(map(int, m.groups()))
                else:
                    if (x := re.findall(r"\d*\.\d+", v)):
                        meta[k] = float(x[0])
                    else:
                        meta[k] = int(v) if v.isdigit() else bool(v)

    train_fn = os.path.join(path, "results")
    if not os.path.exists(train_fn):
        print(f"WARNING: results does not exist in {path}", file=sys.stderr)
    else:
        with open(train_fn, newline=None) as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            for row in reader:
                writer.writerow({**meta, **row})
