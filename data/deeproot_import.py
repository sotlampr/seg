#!/usr/bin/env python
"""
Five seasons with DeepRootLab: A unique facility for easier deep root research
in the field
https://www.biorxiv.org/content/10.1101/2025.04.20.649645v1

Two outputs:
    deeproot_ann: Sparse annotations from RootPainter
    deeproot_seg: Full segmentations (produced by the model?)
"""
import urllib.request
import io
import zipfile
import tempfile
import os
import re

from utils import (
    check_equal_annotations_photos,
    download,
    make_directories,
    zimg_to_disk
)


NAME_ANN = "deeproot_ann"
NAME_SEG = "deeproot_seg"
URL = "https://zenodo.org/api/records/15213661/files-archive"
FNAME = "files-archive.zip"

make_directories(NAME_ANN)
make_directories(NAME_SEG)

train_samples = set()
val_samples = set()
segmentations_samples = dict()

segmentations_todo = set()

# current type
ct = "annotations"

with tempfile.TemporaryFile("w+b") as fo:
    # download_to_stream(URL, fo, 1127)
    download(URL, FNAME, fo, 1127)
    print("Extracting images & annotations", end="", flush=True)
    with zipfile.ZipFile(fo) as zf:
        # open nested zip in-memory
        with zipfile.ZipFile(io.BytesIO(zf.read("projects.zip"))) as fm:
            for entry in fm.filelist:
                if entry.filename.endswith(".png"):
                    path, name = os.path.split(entry.filename)

                    # skip hidden macos files
                    if name.startswith("."):
                        continue

                    img_id, _ = os.path.splitext(name)
                    _, path = os.path.split(path)

                    match path:
                        case "train" | "val":
                            print(".", end="", flush=True)
                            if path == "train":
                                train_samples.add(img_id)
                            else:
                                val_samples.add(img_id)
                            zimg_to_disk(fm, entry, NAME_ANN, path, ct, name)
                        case "segmentations":
                            print(".", end="", flush=True)
                            segmentations_todo.add((entry, img_id, name))

            for (entry, img_id, name) in segmentations_todo:
                print(".", end="", flush=True)
                if img_id in train_samples:
                    segmentations_samples[img_id] = "train"
                    zimg_to_disk(fm, entry, NAME_SEG, "train", ct, name)
                elif img_id in val_samples:
                    segmentations_samples[img_id] = "val"
                    zimg_to_disk(fm, entry, NAME_SEG, "val", ct, name)
                else:
                    # 70/10/30 split
                    i = hash(img_id) % 10
                    if i < 6:
                        segmentations_samples[img_id] = "train"
                        zimg_to_disk(fm, entry, NAME_SEG, "train", ct, name)
                    elif i < 7:
                        segmentations_samples[img_id] = "val"
                        zimg_to_disk(fm, entry, NAME_SEG, "val", ct, name)
                    else:
                        segmentations_samples[img_id] = "test"
                        zimg_to_disk(fm, entry, NAME_SEG, "test", ct, name)

        ct = "photos"
        with zipfile.ZipFile(
            io.BytesIO(zf.read("datasets.zip"))
        ) as fm:
            for entry in fm.filelist:
                if entry.filename.endswith(".jpg"):
                    path, name = os.path.split(entry.filename)

                    if name.startswith("."):
                        continue

                    print(".", end="", flush=True)
                    img_id, _ = os.path.splitext(name)
                    _, path = os.path.split(path)

                    if img_id in train_samples:
                        zimg_to_disk(fm, entry, NAME_ANN, "train", ct, name)
                    elif img_id in val_samples:
                        zimg_to_disk(fm, entry, NAME_ANN, "val", ct, name)

                    if img_id in segmentations_samples:
                        subset = segmentations_samples[img_id]
                        zimg_to_disk(fm, entry, NAME_SEG, subset, ct, name)

print()
print("annotations:")
check_equal_annotations_photos(NAME_ANN)
print("segmentations:")
check_equal_annotations_photos(NAME_SEG)
