#!/usr/bin/env python
"""
Copyright (C) 2025  Sotiris Lamrpinidis

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
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import _apply_op as _apply_augment_op

import \
    m2f, sam, unet, segmentation_pytorch, torchvision_models, mb_sam
from common import IMAGENET_NORM, IMAGENET_MIN

MODULES = [m2f, sam, unet, torchvision_models, segmentation_pytorch, mb_sam]


def all_models():
    for module in MODULES:
        for model in module.models.keys():
            yield (module, model)


def get_patches(input_shape, target_shape):
    """Get overlapping patches in (top, bottom, height, width) format.
    The last patch is overlapping with the second-to-last.
    """
    (m1, n1), (m2, n2) = input_shape, target_shape
    if m1 <= m2 and n1 <= n2:
        yield 0, 0, *(x + 32 - (x % 32) for x in input_shape)
    else:
        for i in range(0, n1, n2):
            for j in range(0, m1, m2):
                i -= max(0, i+n2-n1)
                j -= max(0, j+m2-m1)
                yield i, j, n2, m2


def train(
    model, train_loader, val_loader, checkpoint_dir, epochs,
    lr=1e-3, eval_frequency=80, warmup_steps=200, use_amp=False,
    clip_gradients=False, timeout=None, save_val_images=False,
    extra_val_metrics=False
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr*1e-2)
    lr_log = log10(lr)
    best_score = 0.
    last_update_step = 0
    epoch = 1
    global_step = 1
    patience = 20*eval_frequency
    model.train()

    with open(f"{checkpoint_dir}/results", "a") as fp:
        print(
            "epoch", "step", "wall_duration",
            "train_loss", "val_loss", "val_iou", "val_fscore",
            sep="\t", file=fp
        )

    begin = time.time()
    while True:
        if epoch > epochs:
            print()
            return
        train_loss = 0
        for step, (imgs, msks) in enumerate(train_loader, 1):
            imgs = imgs.to("cuda", non_blocking=True)
            msks = msks.to("cuda", non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(imgs)
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

                    with open(f"{checkpoint_dir}/best", "a") as fp:
                        print(*row, sep="\t", file=fp)

                elif (global_step - last_update_step) > patience:
                    print("\t done")
                    return
                else:
                    print()

                model.train()

                if timeout is not None and time.time() - begin > timeout:
                    print("\t timeout reached.")
                    return

            if global_step <= warmup_steps:
                lr = 10**(lr_log-2*(1-(global_step/warmup_steps)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"  LR is NOW {param_group['lr']:.03g}", end="    ")
            global_step += 1
        train_loss = 0
        epoch += 1


class TrivialAugment(nn.Module):
    def __init__(
        self, num_bins=31, interpolation_mode=InterpolationMode.NEAREST,
        fill=None
    ):
        super().__init__()
        self.fill = fill
        self.interpolation_mode = interpolation_mode
        # op_name: (magnitudes, signed, spatial)
        self.augmentation_space = {
            "Identity": (torch.tensor(0.0), False, False),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True, False),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True, False),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True, False),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True, False),
            "Posterize": (
                8 - (
                    torch.arange(num_bins) / ((num_bins - 1) / 6)
                ).round().int(),
                False, False
            ),
            "Solarize": (
                torch.linspace(255.0, 0.0, num_bins), False, False
            ),
            "AutoContrast": (torch.tensor(0.0), False, False),
            "Equalize": (torch.tensor(0.0), False, False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True, True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True, True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True, True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True, True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True, True),
        }
        self.augmentation_keys = list(self.augmentation_space.keys())
        self.num_augmentations = len(self.augmentation_keys)

    def apply(self, img, op_name: str, magnitude: float):
        return _apply_augment_op(
            img, op_name, magnitude,
            interpolation=self.interpolation_mode,
            fill=self.fill
        )

    def forward(self, img, mask):
        op_index = int(torch.randint(self.num_augmentations, (1,)).item())
        op_name = self.augmentation_keys[op_index]
        magnitudes, signed, spatial = self.augmentation_space[op_name]
        magnitude = (
            float(magnitudes[torch.randint(
                len(magnitudes), (1,), dtype=torch.long
            )].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        if spatial:
            aug = self.apply(torch.cat([img, mask]), op_name, magnitude)
            img, mask = aug[:3], aug[-1:]
        else:
            img = self.apply(img, op_name, magnitude)
        return img, mask


class PlainDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_fnames, mask_fnames, shape=(1024, 1024), augment=False
    ):
        self.normalize = v2.Normalize(**IMAGENET_NORM)

        if augment:
            self.augment = torch.jit.script(TrivialAugment())
        else:
            self.augment = None

        # read image headers and register appropriate crop indices
        self.indices = []
        for (img_fname, mask_fname) in zip(img_fnames, mask_fnames):
            with open(img_fname, "rb") as fp:
                sig = fp.read(4)
                assert sig == b'\x89PNG'
                fp.read(12)
                img_shape = (*map(int.from_bytes, (fp.read(4), fp.read(4))),)
                self.indices.extend(zip(
                    repeat(img_fname), repeat(mask_fname),
                    get_patches(img_shape, shape)
                ))

    def __getitem__(self, idx):
        img_fname, mask_fname, crop_indices = self.indices[idx]

        img = read_image(img_fname)
        img = TF.crop(img, *crop_indices)

        mask = read_image(mask_fname, ImageReadMode.GRAY)
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


def read_data(path, shape):
    return (
        PlainDataset(*map(
            partial(get_img_fnames, path, subset),
            ("photos", "annotations")
        ), shape, augment)
        for subset, augment in (("train", True), ("val", False))
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
    """Expects input as logits"""
    input = (input >= 0).view(-1)
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
    parser.add_argument("-e", "--epochs", type=int, default=80)
    parser.add_argument("-f", "--eval-frequency", type=int, default=200)
    parser.add_argument("-j", "--num-workers", type=int, default=8)
    parser.add_argument("-m", "--mixed-precision", action="store_true")
    parser.add_argument("-o", "--checkpoint-dir", default="../out")
    parser.add_argument(
        "-s", "--shape", type=int, nargs=2, default=(1024, 1024)
    )
    parser.add_argument("-t", "--timeout", type=int)
    parser.add_argument("-w", "--warmup-steps", type=int, default=400)
    parser.add_argument("-C", "--clip-gradients", action="store_true")
    parser.add_argument("-P", "--pretrained", action="store_true")
    parser.add_argument("-S", "--save-val-images", action="store_true")
    parser.add_argument("-X", "--extra-val-metrics", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_dataset, eval_dataset = read_data(args.data_path, args.shape)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=True, multiprocessing_context="fork",
        prefetch_factor=1
    )
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        multiprocessing_context="fork", prefetch_factor=1
    )
    module, model_name = models[args.model]
    model = module.new(model_name, pretrained=args.pretrained).to("cuda")

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    try:
        train(
            model, train_loader, val_loader, args.checkpoint_dir, args.epochs,
            lr=args.learning_rate,
            eval_frequency=args.eval_frequency/args.batch_size,
            warmup_steps=args.warmup_steps/args.batch_size,
            use_amp=args.mixed_precision,
            clip_gradients=args.clip_gradients,
            timeout=args.timeout,
            save_val_images=args.save_val_images,
            extra_val_metrics=args.extra_val_metrics
        )
    except KeyboardInterrupt:
        print()
        print("Received keyboard interrupt. bye!")

    mem = torch.cuda.memory_reserved()
    print(f"GPU: {mem/1024**3:.1f} GB reserved")
    sys.exit(0)
