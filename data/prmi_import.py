#!/usr/bin/env python
"""
TODO: description
"""
import urllib.request
import io
import zipfile
import tempfile
import os
import re

from utils import (
    check_equal_annotations_photos,
    download_to_stream,
    make_directories,
    as_png
)

# NAME = "grassland"
URL = "https://datadryad.org/downloads/file_stream/1319529"

# make_directories(NAME)

datasets = [
    "Cotton_736x552_DPI150",
    "Papaya_736x552_DPI150",
    "Peanut_736x552_DPI150",
    "Sesame_736x552_DPI150",
    "Sunflower_640x480_DPI120",
    "Switchgrass_720x510_DPI300",
]

regexp = "".join((
    "^PRMI_official/(train|val|test)/(images|masks_pixel_gt)/",
    fr"({'|'.join(datasets)})/.*\.(png|jpg)"
))

dataset_names = [x.split("_", maxsplit=1)[0].lower() for x in datasets]
for name in dataset_names:
    make_directories(name)

with tempfile.TemporaryFile("w+b") as fo:
    print("Downloading...")
    download_to_stream(URL, fo, 9296)
    print("Extracting images & annotations", end="", flush=True)
    with zipfile.ZipFile(fo) as zf:
        for entry in zf.filelist:
            if m := re.match(regexp, entry.filename):
                subset, kind, dataset, filetype = m.groups()
                _, name = os.path.split(entry.filename)

                dataset_name = dataset.split("_", maxsplit=1)[0].lower()
                path = "photos" if kind == "images" else "annotations"
                out = os.path.join(dataset_name, subset, path, name)
                print("\r", out, sep="", end="", flush=True)

                with open(out, "wb") as f:
                    #if filetype == "jpg":
                    #    as_png(io.BytesIO(zf.read(entry.filename)), f)
                    #else:
                    f.write(zf.read(entry.filename))

print()
for name in dataset_names:
    print(f"\t{name}")
    check_equal_annotations_photos(name)
