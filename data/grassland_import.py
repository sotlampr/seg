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
    download,
    make_directories
)

NAME = "grassland"
URL = "https://figshare.com/ndownloader/articles/20440497/versions/2"
FNAME= "20440497.zip"

make_directories(NAME)


with tempfile.TemporaryFile("w+b") as fo:
    download(URL, FNAME, fo, 957)
    print("Extracting images & annotations", end="", flush=True)
    with zipfile.ZipFile(fo) as zf:
        # open nested zip in-memory
        with zipfile.ZipFile(io.BytesIO(zf.read("CNN_model.zip"))) as fm:
            for entry in fm.filelist:
                if entry.filename.endswith(".png"):
                    path, name = os.path.split(entry.filename)
                    _, path = os.path.split(path)

                    # skip hidden macos files
                    if name.startswith("."):
                        continue

                    print(".", end="", flush=True)
                    out = os.path.join(NAME, "train", path, name)
                    with open(out, "wb") as f:
                        f.write(fm.read(entry.filename))

        with zipfile.ZipFile(
            io.BytesIO(zf.read("CNN_validation_Images.zip"))
        ) as fm:
            for entry in fm.filelist:
                if entry.filename.endswith(".png"):
                    path, name = os.path.split(entry.filename)
                    _, path = os.path.split(path)

                    # skip hidden macos files
                    if name.startswith("."):
                        continue

                    print(".", end="", flush=True)
                    out = os.path.join(NAME, "val", path, name)
                    with open(out, "wb") as f:
                        f.write(fm.read(entry.filename))

print()
check_equal_annotations_photos(NAME)
