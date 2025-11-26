#!/usr/bin/env python3
import argparse
import csv
import glob
import os.path
import re
import sys

import numpy as np
import pandas as pd
import scipy.stats


def parse_num(x):
    if re.match(r"^\d+\.\d+$", x):
        return float(x)
    elif x.lower() in {"na", "nan"}:
        return float("nan")
    elif x.isdigit():
        return int(x)
    else:
        return x


def key_value_pairs(*args):
    if len(args) == 1:
        key, val = args[0].split("=")
    else:
        raise Exception("Use -X key=val")
    return key, val


parser = argparse.ArgumentParser()
parser.add_argument("base_path", help="Should contain [DATASET]/[MODEL]")
parser.add_argument("-X", "--constant", action="append",
                    default=list(), type=key_value_pairs)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-S", "--skip-header", action="store_true")
parser.add_argument("-C", "--convert", action="store_true", help="Convert to mm")
args = parser.parse_args()
args.constant = dict(args.constant)

unit = "mm" if args.convert else "px"

# keys to extract from rv csv file, and our own name
rv_keys = {
    **{
        f"Total.Root.Length.{unit}": f"total_root_length_{unit}",
        f"Average.Diameter.{unit}": f"average_diameter_{unit}"
    },
    **{
        k.format(i, unit): v.format(i, unit)
        for k, v in [
            ("Root.Length.Diameter.Range.{}.{}", "root_length_diameter_bin_{}_{}")
        ]
        for i in range(1, 6)
    }
}

conversion_factors = {
    "chicory": 13.303,
    "cotton": 5.906,
    "deeproot_ann": 51.2,
    "grassland": 47.244,
    "papaya": 5.906,
    "peanut": 5.906,
    "sesame": 5.906,
    "sunflower": 4.724,
    "switchgrass": 11.811
}

directory_regexp = r"(?P<model>\S+)-(?P<dataset>\S+?)-(?P<run>[0-9])-(?P<pretrained>(?:pretrained|scratch))?$"  # noqa: E501

datasets = dict()
models = dict()
for dn in glob.glob(os.path.join(args.base_path, "*/")):
    # Read the target segmentation features from annotations
    if args.verbose:
        print(f"reading {dn}...", file=sys.stderr)
    dataset = os.path.basename(os.path.dirname(dn))
    datasets[dataset] = {
        "index": dict(),
        **{key: list() for key in rv_keys.values()}
    }

    rv_fn = os.path.join(dn, "annotations", "features.csv")
    if not os.path.exists(rv_fn):
        print(f"WARNING: features.csv does not exist in {dn}/annotations",
              file=sys.stderr)
    else:
        with open(rv_fn, newline=None) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                fn = row["File.Name"]
                datasets[dataset]["index"][fn] = \
                    len(datasets[dataset]["index"])
                for k1, k2 in rv_keys.items():
                    datasets[dataset][k2].append(parse_num(row[k1]))

        for key in rv_keys.values():
            datasets[dataset][key] = np.array(datasets[dataset][key])

    # Go through each model subfolder
    for mn in glob.glob(os.path.join(dn, "*/")):
        if args.verbose:
            print(f"reading {mn}...", file=sys.stderr)
        model = os.path.basename(os.path.dirname(mn))
        if model in {"annotations", "photos"}:
            continue
        models[model] = re.match(directory_regexp, model).groupdict()

        # Get rhizovision features
        rv_fn = os.path.join(mn, "features.csv")
        if not os.path.exists(rv_fn):
            print(f"WARNING: features.csv does not exist in {mn}",
                  file=sys.stderr)
        else:

            for k in rv_keys.values():
                models[model][k] = np.full_like(datasets[dataset][k], np.nan)

            with open(rv_fn, newline=None) as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    fn = row["File.Name"]
                    idx = datasets[dataset]["index"][fn]
                    for k1, k2 in rv_keys.items():
                        models[model][k2][idx] = parse_num(row[k1])

        # Get F1 scores
        fscore_fn = os.path.join(mn, "f1_score.csv")
        if not os.path.exists(fscore_fn):
            print(f"WARNING: f1_score.csv does not exist in {mn}",
                  file=sys.stderr)
        else:
            models[model]["f1_score"] = \
                np.full((len(datasets[dataset]["index"]),), np.nan)

            with open(fscore_fn, newline=None) as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    if row["filename"] == "micro":
                        models[model]["f1_micro"] = float(row["f1_score"])
                        continue
                    idx = datasets[dataset]["index"][row["filename"]]

                    models[model]["f1_score"][idx] = float(row["f1_score"])

if args.convert:
    for dname, dataset in datasets.items():
        for key in list(dataset.keys()):
            if key.endswith("_px") and args.convert:
                dataset[key[:-3] + "_mm"] = \
                    dataset[key] / conversion_factors[dname]

    for model in models.values():
        for key in list(model.keys()):
            if key.endswith("_px") and args.convert:
                model[key[:-3] + "_mm"] = \
                    model[key] / conversion_factors[model["dataset"]]
                del model[key]

# Create output "table"
results = []
for model_id, model in sorted(models.items()):
    targets = datasets[model["dataset"]]

    f1s = model["f1_score"]
    f1s = f1s[f1s != 0]

    out = {**args.constant, **model,
           "f1_macro": np.nanmean(f1s)}
    del out["f1_score"]

    for key in rv_keys.values():
        if key.endswith("_px") and args.convert:
            key = key[:-3] + "_mm"

        if key not in model:
            print(f"WARNING: {key} not in {model_id}", file=sys.stderr)
            if "_bin_" not in key:
                out[f"{key}_pearsonr_statistic"] = np.nan
                out[f"{key}_pearsonr_pvalue"] = np.nan
                out[f"{key}_mean_absolute_error"] = np.nan
            out[f"{key}_sum"] = np.nan
            out[f"{key}_mean"] = np.nan
            continue

        x = model[key]
        y = targets[key]
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0

        if "_bin_" not in key:
            corr = scipy.stats.pearsonr(x, y)
            out[f"{key}_pearsonr_statistic"] = float(corr.statistic)
            out[f"{key}_pearsonr_pvalue"] = float(corr.pvalue)
            out[f"{key}_mean_absolute_error"] = float(np.abs(x-y).mean())

        out[f"{key}_sum"] = x.sum()
        out[f"{key}_mean"] = x.mean()
        del out[key]

    results.append(out)

if not args.skip_header:
    for dataset_name, dataset in datasets.items():
        out = {"model": "gold_annotation", "dataset": dataset_name}
        for key in rv_keys.values():
            if key.endswith("_px") and args.convert:
                key = key[:-3] + "_mm"
            x = dataset[key]
            out[f"{key}_sum"] = x.sum()
            out[f"{key}_mean"] = x.mean()
        results.append(out)
        
df = pd.DataFrame.from_dict(results)
df.to_csv(sys.stdout, header=not args.skip_header, index=False)
