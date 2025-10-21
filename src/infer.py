#!/usr/bin/env python
"""
Copyright (C) 2025  Sotiris Lamprinidis, Abraham Smith

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import argparse
from functools import partial, lru_cache
import math
import os
import random
import sys
import traceback

import torch
from torchvision.io import decode_image, write_png, write_jpeg
from torchvision.transforms.v2.functional import \
    center_crop, convert_image_dtype, normalize, pad

from common import \
    IMAGENET_NORM, all_models, expand_filename, load_model, torch_init


def path_for(dataset, base_path, subset="val"):
    return os.path.join(base_path, dataset, subset)


def make_path(fname, ident, dataset, base_path):
    if ident:
        base_fn, ext = os.path.splitext(fname)
        rest = (ident, fname)
    else:
        rest = (fname,)
    return os.path.join(base_path, dataset, *rest)


def make_dirs(dataset, base_path=None, *paths):
    os.makedirs(os.path.join(base_path, dataset), exist_ok=True)
    for path in paths:
        os.makedirs(os.path.join(base_path, dataset, path), exist_ok=True)


@lru_cache
def get_images(dataset, base_path, out_path, subset="val", num_samples=None, seed=42):
    """ Read images and annotations for a dataset, cache, and save them
        to out_path.
    """
    print("loading", dataset)
    make_path_f = partial(make_path, base_path=out_path, dataset=dataset)
    data_path = path_for(dataset, base_path, subset)
    fns = list(zip(*map(
        lambda x: sorted(map(partial(os.path.join, x), os.listdir(x))),
        map(partial(os.path.join, data_path), ("photos", "annotations"))
    )))

    if num_samples is not None:
        random.seed(seed)
        fns = random.choices(fns, k=num_samples)

    imgs, masks = zip(*map(lambda a: tuple(map(decode_image, a)), fns))

    if masks[0].ndim == 3 and masks[0].shape[0] == 4:
        masks = [m[:-1] for m in masks]

    make_dirs(dataset, out_path, "photos", "annotations")
    for i, (img, mask, (img_fn, ann_fn)) in enumerate(zip(imgs, masks, fns)):
        if img_fn.endswith(".jpg") or img_fn.endswith(".jpeg"):
            write_jpeg(
                convert_image_dtype(img, torch.uint8),
                make_path_f(os.path.split(img_fn)[1], "photos")
            )
        else:
            write_png(
                convert_image_dtype(img, torch.uint8),
                make_path_f(os.path.split(img_fn)[1], "photos")
            )
        write_png(
            convert_image_dtype(mask, torch.uint8),
            make_path_f(os.path.split(ann_fn)[1], "annotations")
        )

    imgs = convert_image_dtype(torch.stack(imgs), torch.float)
    masks = torch.stack(masks).any(1).to(torch.float)

    return imgs, masks, fns


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
    if (not args.force) \
            and (os.path.exists(args.out_path)
                 and len(os.listdir(args.out_path)) != 0):
        print(
            f"-f not given and '{args.out_path}' exists or not empty, exiting")
        return 1

    os.makedirs(args.out_path, exist_ok=True)
    get_images_f = partial(
        get_images, subset=args.subset, base_path=args.data_path,
        out_path=args.out_path, num_samples=args.num_samples)
    models = {f"{k.__name__}/{v}": (k, v) for k, v in all_models()}

    for model_fn in args.model_checkpoints:
        print("doing", model_fn)
        run = expand_filename(model_fn)

        model_name = os.path.split(os.path.split(model_fn)[0])[1]
        if not model_name.endswith("pretrained"):
            model_name += "-scratch"
        make_dirs(run["dataset"], args.out_path, model_name)
        make_path_f = partial(
            make_path, dataset=run["dataset"], base_path=args.out_path,
        )
        model_id = f"{run['package']}/{run['model']}"

        if run["attributes"]:
            model_id += '-' + run["attributes"]

        model = load_model(
             model_id, pretrained=run["pretrained"], models=models
        ).to(args.device).eval()

        try:
            model.load_state_dict(
                torch.load(model_fn, weights_only=False), strict=False)
        except RuntimeError:
            traceback.print_exc()
            continue

        # get input shape for this model from '$path/config' file
        config_fn = os.path.join(os.path.split(model_fn)[0], "config")
        in_shape = None
        with open(config_fn) as fp:
            for ln in fp.readlines():
                key, val = ln.strip().split("\t", maxsplit=1)
                if key == "shape":
                    in_shape = tuple(map(int, val.strip("[]").split(",")))
                    break

        assert in_shape is not None
        imgs, annotations, fns = get_images_f(run["dataset"])

        # run a sample inference to get output shape
        model.eval()
        with torch.no_grad(), torch.inference_mode():
            sample = model(torch.zeros((1, 3, *in_shape)).to(args.device))
            out_shape = sample.shape[2:]

        for i, (img, (_, fn)) in enumerate(zip(imgs, fns), 1):
            print(f"\r{i:03d}/{len(imgs)}", end="")
            segmented = segment(model, img, in_shape, out_shape, args.device)
            write_png(
                convert_image_dtype(segmented.unsqueeze(0), torch.uint8),
                make_path_f(os.path.split(fn)[1], model_name))
        print(79*" ", "\r", end="")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoints", nargs="+")
    parser.add_argument("-d", "--data-path", default="../data")
    parser.add_argument("-o", "--out-path", default="../inferred")
    parser.add_argument("-s", "--subset", default="val", choices=("train", "val", "test"))
    parser.add_argument("-n", "--num-samples", type=int)
    parser.add_argument("-D", "--device", default="cuda")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    torch_init(deterministic=True)

    sys.exit(main(args))
