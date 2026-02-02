#!/usr/bin/env python
"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis, Abraham Smith

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
----------
Infer segmentations for one or more models.
The models are assumed to reside in a directory matching
($PACKAGE)-($MODEL_VARIANT)-($DATASET)-($SEED)(-pretrained)?
"""
import argparse
from functools import partial
import glob
import math
import os
import random
import sys
import traceback

import torch
from torchvision.io import decode_image, write_png
from torchvision.transforms.v2.functional import \
    center_crop, convert_image_dtype, normalize, pad

from seg_common import \
    DATA_ROOT, IMAGENET_NORM, \
    get_model_dict, load_model, torch_init


def path_for(dataset, base_path, subset="val"):
    return os.path.join(base_path, dataset, subset)


def get_images(dataset, base_path, out_path, subset="val", num_samples=None,
               seed=42):
    """ Read images and annotations for a dataset, cache, and save them
        to out_path.
    """
    print("loading", dataset)
    data_path = path_for(dataset, base_path, subset)
    fns = list(zip(*map(
        lambda x: sorted(
            map(partial(os.path.join, x), glob.glob(f"{x}/*.[jp][pn]g"))),
        map(partial(os.path.join, data_path), ("photos", "annotations"))
    )))

    if num_samples is not None:
        random.seed(seed)
        fns = random.choices(fns, k=num_samples)

    data_iter = map(lambda a: tuple(map(decode_image, a)), fns)

    for i, ((img, mask), (img_fn, ann_fn)) in enumerate(zip(data_iter, fns)):
        if mask.ndim == 3 and mask.shape == 4:
            mask = mask[:-1]
        img = convert_image_dtype(img, torch.float)
        mask = mask.any(1).to(torch.float)
        yield img, mask, (img_fn, ann_fn)


def tiles_from_coords(image, coords, tile_shape):
    # image is (C, H, W)
    tiles = []
    for x, y in coords:
        tile = image[:, y:y+tile_shape[0], x:x+tile_shape[1]]
        tiles.append(tile)
    return tiles


def get_tiles(image, in_tile_shape, out_tile_shape):
    # image is (C, H, W)
    if image.shape[1] <= in_tile_shape[0] \
            and image.shape[2] <= in_tile_shape[1]:
        # nothing needed
        return [image], [(0, 0)]

    pad_h = max(0, in_tile_shape[0] - out_tile_shape[0]) // 2
    pad_w = max(0, in_tile_shape[1] - out_tile_shape[1]) // 2
    if pad_h or pad_w:
        image = pad(image, (pad_w, pad_h, pad_w, pad_h))

    horizontal_count = math.ceil(image.shape[2] / out_tile_shape[1])
    vertical_count = math.ceil(image.shape[1] / out_tile_shape[0])

    # first split the image based on the tiles that fit
    x_coords = [h*out_tile_shape[1] for h in range(horizontal_count-1)]
    y_coords = [v*out_tile_shape[0] for v in range(vertical_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile position by subtracting tile size from the
    # edge of the image.
    right_x = max(0, image.shape[2] - in_tile_shape[1])
    bottom_y = max(0, image.shape[1] - in_tile_shape[0])

    x_coords.append(right_x)
    y_coords.append(bottom_y)

    # because its a rectangle get all combinations of x and y
    tile_coords = [(x, y) for x in x_coords for y in y_coords]
    tiles = tiles_from_coords(image, tile_coords, in_tile_shape)
    return tiles, tile_coords


def reconstruct_from_tiles(tiles, coords, output_shape):
    image = torch.zeros(output_shape)
    for tile, (x, y) in zip(tiles, coords):
        image[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return image


def segment(model, image, in_shape, out_shape, device):
    tiles, coords = get_tiles(
        image, in_tile_shape=in_shape, out_tile_shape=out_shape
    )
    with torch.no_grad(), torch.inference_mode():
        output_tiles = []
        for i, tile in enumerate(tiles):
            orig_shape = None
            if in_shape != tile.shape[1:] and in_shape == out_shape:
                # pad image
                orig_shape = tile.shape[1:]
                tile = center_crop(tile, in_shape)
            tile = normalize(tile.to(device), **IMAGENET_NORM).unsqueeze(0)
            output = model(tile)
            if orig_shape:
                # crop image
                output = center_crop(output, orig_shape)
            elif output.shape[2:] == (484, 500):
                # hack for unet/GNRes
                output = center_crop(output, (480, 500))

            predicted = (output >= 0).squeeze(0).squeeze(0)
            output_tiles.append(predicted)

    reconstructed = reconstruct_from_tiles(
        output_tiles, coords, image.shape[1:])
    return reconstructed


def main(args):
    torch_init(deterministic=True)

    if (not args.force) \
            and (os.path.exists(args.out_path)
                 and len(os.listdir(args.out_path)) != 0):
        print(
            f"-f not given and '{args.out_path}' exists or not empty, exiting")
        return 1

    os.makedirs(args.out_path, exist_ok=True)
    get_images_f = partial(
        get_images, subset=args.subset, base_path=DATA_ROOT,
        out_path=args.out_path, num_samples=args.num_samples)

    models = get_model_dict()

    model_fn = args.model_checkpoint

    # get input shape for this model from '$path/config' file
    config_fn = os.path.join(os.path.dirname(model_fn), "config")
    in_shape, model_type = None, None
    with open(config_fn) as fp:
        for ln in fp.readlines():
            key, val = ln.strip().split("\t", maxsplit=1)
            if key == "shape":
                in_shape = tuple(map(int, val.strip("[]").split(",")))
            elif key == "model":
                model_type = val
            if in_shape is not None and model_type is not None:
                break

    model = \
        load_model(model_type, False, True, models).to(args.device).eval()
    weights = torch.load(model_fn, weights_only=False)

    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print(f"FAILED: {model_fn}")
        traceback.print_exc()
        print("Loading with strict=false")
        model.load_state_dict(weights, strict=False)

    # run a sample inference to get output shape
    model.eval()
    with torch.no_grad(), torch.inference_mode():
        sample = model(torch.zeros((1, 3, *in_shape)).to(args.device))
        out_shape = sample.shape[2:]

    im_iter = get_images_f(args.dataset)
    for i, (img, _, (_, fn)) in enumerate(im_iter, 1):
        print(f"\r{i:04d}", end="")
        segmented = \
            segment(model, img, in_shape, out_shape, args.device)
        write_png(
            convert_image_dtype(segmented.unsqueeze(0), torch.uint8),
            os.path.join(args.out_path, os.path.basename(fn)))
        print(79*" ", "\r", end="")
    return 0


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("-o", "--out-path", default="../inferred")
    parser.add_argument("-s", "--subset", default="val",
                        choices=("train", "val", "test"))
    parser.add_argument("-n", "--num-samples", type=int)
    parser.add_argument("-D", "--device", default="cuda")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()
    return main(args)


if __name__ == "__main__":
    sys.exit(cli_main())
