#!/usr/bin/env python
"""
Copyright (C) 2025  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import argparse
from functools import partial
import glob
from itertools import repeat
from math import log10
import os
import re
import sys
import time

import torch
from torch import nn
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as v2
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2.functional as TF

from common import IMAGENET_NORM, all_models, load_model, torch_init


def get_patches(input_shape, target_shape):
    """Get overlapping patches in (top, bottom, height, width) format.
    The last patch is overlapping with the second-to-last.
    """
    (m1, n1), (m2, n2) = input_shape, target_shape
    if m1 <= m2 and n1 <= n2:
        yield 0, 0, *(x + 32 - (x % 32) for x in input_shape)
    else:
        for i in range(0, m1, m2):
            for j in range(0, n1, n2):
                i -= max(0, i+m2-m1)
                j -= max(0, j+n2-n1)
                yield i, j, m2, n2


def train(
    model, train_loader, val_loader, checkpoint_dir, epochs,
    lr=1e-3, eval_frequency=80, warmup_steps=200, use_amp=False,
    clip_gradients=False, timeout=None, save_val_images=False,
    patience=20, extra_val_metrics=False, sparse_annotations=False
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr*1e-2)
    lr_log = log10(lr)
    best_score = 0.
    last_update_step = 0
    epoch = 1
    global_step = 1
    patience = patience * eval_frequency
    model.train()
    state = ("running", "initializing")

    with open(f"{checkpoint_dir}/results", "w") as fp:
        print(
            "epoch", "step", "wall_duration",
            "train_loss", "val_loss", "val_iou", "val_fscore",
            sep="\t", file=fp
        )

    begin = time.time()
    while True and state[0] == "running":
        state = ("running", "training")
        if epoch > epochs:
            state = ("done", "epoch limit reached")
            print()
            break
        train_loss = 0
        for step, (imgs, msks) in enumerate(train_loader, 1):
            imgs = imgs.to("cuda", non_blocking=True)
            msks = msks.to("cuda", non_blocking=True)

            if msks.any():
                with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred = model(imgs)
                    if pred.shape[2:] != imgs.shape[2:]:
                        msks = TF.center_crop(msks, pred.shape[2:])
                    if sparse_annotations:
                        keep = msks[:, :2].any(1)  # either red or green
                        pred = pred[keep.unsqueeze(1)]
                        msks = msks[:, 0][keep]
                    loss = dice_loss(pred, msks) \
                        + F.binary_cross_entropy_with_logits(pred, msks)

                if loss.isnan():
                    print("  nan loss!!!")
                    continue

                train_loss += loss.item()
                print(
                    f"\r{epoch:3d} {global_step:6d} {train_loss/step:.03f} "
                    f"{loss.cpu().item():.3f}", end=""
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_gradients:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if global_step % eval_frequency == 0:
                eval_loss = 0
                iou = 0
                tps, fps, fns = 0, 0, 0
                model.eval()
                for i, (imgs, msks) in enumerate(val_loader):
                    imgs = imgs.to("cuda", non_blocking=True)
                    msks = msks.to("cuda", non_blocking=True)
                    with autocast(
                        "cuda", dtype=torch.bfloat16, enabled=use_amp
                    ):
                        with torch.inference_mode(), torch.no_grad():
                            pred = model(imgs)
                            if pred.shape[2:] != imgs.shape[2:]:
                                msks = TF.center_crop(msks, pred.shape[2:])

                            if sparse_annotations:
                                keep = msks[:, :2].any(1)
                                pred = pred[keep.unsqueeze(1)]
                                msks = msks[:, 0][keep]

                            if extra_val_metrics:
                                loss = dice_loss(pred, msks) \
                                    + F.binary_cross_entropy_with_logits(
                                        pred, msks
                                    )

                    if extra_val_metrics:
                        eval_loss += loss.item()
                        iou += get_iou(pred, msks).item()
                    tp, fp, fn = tp_fp_fn(pred, msks)
                    tps += tp
                    fps += fp
                    fns += fn

                if extra_val_metrics:
                    eval_loss /= len(val_loader)
                    iou /= len(val_loader)
                fscore = f1_score_(tps, fps, fns).item()
                duration = time.time()-begin

                print(
                    f"\r{epoch:3d} {global_step:6d} {duration:3.1f}s "
                    f"train: {train_loss/step:.03f} val: {eval_loss:.03f} "
                    f"iou: {iou:.03f} f1: {fscore:.03f} ", end=""
                )

                row = (
                    epoch, global_step, duration, train_loss/step,
                    eval_loss, iou, fscore
                )
                with open(f"{checkpoint_dir}/results", "a") as fp:
                    print(*row, sep="\t", file=fp)

                if fscore > best_score:
                    best_score = fscore
                    last_update_step = global_step
                    print('\tBEST')
                    torch.save(
                        model.state_dict(),
                        checkpoint_dir + "/checkpoint_best.pth"
                    )
                    if save_val_images:
                        for k in range(max(imgs.size(0), 3)):
                            TF.to_pil_image(
                                imgs[k]
                            ).save(f"{checkpoint_dir}/{k}-img.png")
                            TF.to_pil_image(
                                F.sigmoid(pred[k])
                            ).save(f"{checkpoint_dir}/{k}-pred.png")
                            TF.to_pil_image(
                                msks[k]
                            ).save(f"{checkpoint_dir}/{k}-true.png")

                    with open(f"{checkpoint_dir}/best", "w") as fp:
                        print(*row, sep="\t", file=fp)
                elif (global_step - last_update_step) > patience:
                    print("\t done")
                    state = ("done", "ok")
                    break
                else:
                    print()

                model.train()

                if timeout is not None and time.time() - begin > timeout:
                    print("\t timeout reached.")
                    state = ("done", "timeout reached")
                    break

            if global_step <= warmup_steps:
                lr = 10**(lr_log-2*(1-(global_step/warmup_steps)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            global_step += 1
        train_loss = 0
        epoch += 1
    return state


class OurAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.augment = nn.Sequential(
            v2.RandomApply(torch.nn.ModuleList([
                v2.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
                )]), p=2/3
            ),
            v2.RandomApply(torch.nn.ModuleList([v2.Grayscale(3)]), p=5e-2),
            v2.RandomInvert(p=2.5e-2),
        )
        self.augment_spatial = nn.Sequential(
            v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomRotation(15),
                ]), p=1/3
            ),
            v2.RandomApply(torch.nn.ModuleList([
                    v2.RandomPerspective(distortion_scale=0.4, p=1.0),
                ]), p=1/3
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(torch.nn.ModuleList([
                v2.ElasticTransform(alpha=75.)]), p=1/3
            )
        )

    def forward(self, img, mask):
        img = self.augment(img)
        aug = self.augment_spatial(torch.cat([img, mask]))
        img, mask = aug[:3], aug[3:]
        return img, mask


def get_resolution(fp):
    sig = fp.read(4)
    if sig == b'\x89PNG':
        fp.read(12)
        width = int.from_bytes(fp.read(4))
        height = int.from_bytes(fp.read(4))
    elif sig.startswith(b'\xff\xd8'):
        x = re.search(b"\xff\xc0...(..)(..)", fp.read())
        assert x is not None
        height, width = (int.from_bytes(y) for y in x.groups())
    else:
        raise NotImplementedError(f"signature: {sig}")
    return (height, width)


class PlainDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_fnames, mask_fnames, shape=(1024, 1024),
        sparse_annotations=False, train=False
    ):
        self.normalize = v2.Normalize(**IMAGENET_NORM)

        if train:
            self.augment = torch.jit.script(OurAugment())
        else:
            self.augment = None

        # read image headers and register appropriate crop indices
        self.indices = []
        for (img_fname, mask_fname) in zip(img_fnames, mask_fnames):
            with open(img_fname, "rb") as fp:
                this_shape = get_resolution(fp)

                self.indices.extend(zip(
                    repeat(img_fname), repeat(mask_fname),
                    get_patches(this_shape, shape)
                ))

        if sparse_annotations:
            self.mask_read_mode = ImageReadMode.RGB
        else:
            self.mask_read_mode = ImageReadMode.GRAY

    def __getitem__(self, idx):
        img_fname, mask_fname, crop_indices = self.indices[idx]

        img = read_image(img_fname)
        img = TF.crop(img, *crop_indices)

        mask = read_image(mask_fname, self.mask_read_mode)
        mask = TF.crop(mask, *crop_indices)
        mask = (mask.to(torch.uint8) > 0).to(torch.uint8)

        if self.augment is not None:
            img, mask = self.augment(img, mask)

        img = TF.convert_image_dtype(img, torch.float32)
        img = self.normalize(img)
        mask = mask.to(torch.float32)
        return img, mask

    def __len__(self):
        return len(self.indices)


def get_img_fnames(base, subset, kind):
    path = os.path.join(os.path.join(base, subset), kind)
    return sorted(glob.glob(f"{path}/*"))


def read_data(path, shape, sparse_annotations):
    return (
        PlainDataset(*map(
            partial(get_img_fnames, path, subset),
            ("photos", "annotations")
        ), shape, sparse_annotations, subset == "train")
        for subset in ("train", "val")
    )


def intersection_union(input, target):
    """ based on loss function from V-Net paper """
    input = input.view(-1)
    target = target.reshape(-1)
    intersection = torch.sum(torch.mul(input, target))
    union = torch.sum(input) + torch.sum(target)
    return intersection, union


def dice_loss(input, target, eps=torch.finfo(torch.bfloat16).eps):
    intersection, union = intersection_union(F.sigmoid(input), target)
    return (1 - ((2 * intersection) + eps) / (union + eps))


def get_iou(input, target, eps=torch.finfo(torch.bfloat16).eps):
    intersection, union = intersection_union(input >= 0, target)
    return (intersection + eps) / (union + eps)


def tp_fp_fn(input, target):
    if input.dtype != torch.bool:
        # assume logits
        input = input >= 0
    input = input.view(-1)
    target = target.bool().view(-1)
    tp = (input & target).sum()
    fp = (input & ~target).sum()
    fn = (~input & target).sum()
    return tp, fp, fn


def f1_score_(tp, fp, fn):
    tp *= 2
    return tp / (tp+fp+fn)


def f1_score(input, target):
    return f1_score_(*tp_fp_fn(input, target))


if __name__ == "__main__":
    models = {f"{k.__name__}/{v}": (k, v) for k, v in all_models()}

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model", choices=models)
    parser.add_argument("-a", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-b", "--batch-size", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    parser.add_argument("-f", "--eval-frequency", type=int)
    parser.add_argument("-j", "--num-workers", type=int, default=8)
    parser.add_argument("-m", "--mixed-precision", action="store_true")
    parser.add_argument("-p", "--patience", type=int, default=20)
    parser.add_argument("-o", "--checkpoint-dir", default="../out")
    parser.add_argument(
        "-s", "--shape", type=int, nargs=2, default=(1024, 1024)
    )
    parser.add_argument("-t", "--timeout", type=int)
    parser.add_argument("-w", "--warmup-steps-mul", type=int, default=2)
    parser.add_argument("-A", "--sparse-annotations", action="store_true")
    parser.add_argument("-C", "--clip-gradients", action="store_true")
    parser.add_argument("-N", "--no-optimizations", action="store_true")
    parser.add_argument("-P", "--pretrained", action="store_true")
    parser.add_argument("-S", "--save-val-images", action="store_true")
    parser.add_argument("-X", "--extra-val-metrics", action="store_true")
    parser.add_argument("-Z", default="")
    args = parser.parse_args()

    torch_init()

    train_dataset, eval_dataset = read_data(
        args.data_path, args.shape, args.sparse_annotations)
    print(f"#train: {len(train_dataset)} #val: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        multiprocessing_context="fork" if args.num_workers else None,
        prefetch_factor=1 if args.num_workers else None
    )
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        multiprocessing_context="fork" if args.num_workers else None,
        prefetch_factor=1 if args.num_workers else None
    )

    model = load_model(
        args.model, pretrained=args.pretrained,
        optimize=not args.no_optimizations, models=models
    ).to("cuda")

    if not args.eval_frequency:
        args.eval_frequency = len(train_loader.dataset)

    eval_freq = args.eval_frequency // args.batch_size
    # Known bug
    # if args.eval_frequency % args.batch_size:
    #     eval_freq += 1
    warmup_st = args.warmup_steps_mul * eval_freq

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    with open(f"{args.checkpoint_dir}/config", "w") as fp:
        for k, v in (
            ("batch_size", args.batch_size),
            ("clip_gradents", args.clip_gradients),
            ("eval_frequency", eval_freq),
            ("learning_rate", args.learning_rate),
            ("mixed_precision", args.mixed_precision),
            ("num_epochs", args.epochs),
            ("num_train", len(train_loader.dataset)),
            ("num_val", len(val_loader.dataset)),
            ("optimize", not args.no_optimizations),
            ("patience", args.patience),
            ("pretrained", args.pretrained),
            ("shape", args.shape),
            ("timeout", args.timeout),
            ("warmup_steps", warmup_st),
        ):
            print(k, v, sep="\t", file=fp)

    try:
        status, reason = train(
            model, train_loader, val_loader, args.checkpoint_dir, args.epochs,
            lr=args.learning_rate,
            eval_frequency=eval_freq,
            warmup_steps=warmup_st,
            use_amp=args.mixed_precision,
            clip_gradients=args.clip_gradients,
            timeout=args.timeout,
            patience=args.patience,
            save_val_images=args.save_val_images,
            extra_val_metrics=args.extra_val_metrics,
            sparse_annotations=args.sparse_annotations
        )
    except KeyboardInterrupt:
        print()
        print("Received keyboard interrupt. bye!")
        status, reason = "done", "interrupt"

    mem = torch.cuda.memory_reserved()
    print(f"\nSTATUS: {status}: {reason}")
    print(f"GPU: {mem/1024**3:.1f} GB reserved")
    if status != "done":
        sys.exit(1)
    sys.exit(0)
