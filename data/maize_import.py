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
    make_directories
)

NAME = "maize"

make_directories(NAME)

URL = "https://zenodo.org/records/8224956/files/Banet%20et%20al_Crop%20Root%20Image%20Domain%20Shifts.zip?download=1"  # noqa: E501
with tempfile.TemporaryFile("w+b") as fo:
    download_to_stream(URL, fo)
    print("Extracting images & annotations", end="", flush=True)
    with zipfile.ZipFile(fo) as zf:
        for entry in zf.filelist:
            print(entry.filename)
            if entry.filename.endswith(".jpeg"):
                path, name = os.path.split(entry.filename)
                _, path = os.path.split(path)

                # skip hidden macos files
                if name.startswith("."):
                    continue

                continue
                import ipdb; ipdb.set_trace()
                print(".", end="", flush=True)

                # ensure the same name so hashing works
                # for both pics and annotations
                name = re.sub("-binary", "", name)

                # use the hash to determine subset
                hs = hash(name)
                subset = (
                    "test" if hs % 6 == 0
                    else "val" if hs % 3 == 0
                    else "train"
                )
                out = os.path.join(NAME, subset, path, name)
                with open(out, "wb") as f:
                    f.write(fm.read(entry.filename))

check_equal_annotations_photos(NAME)
