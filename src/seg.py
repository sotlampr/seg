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
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as v2
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2.functional as TF

import \
    m2f, sam, unet, segmentation_pytorch, torchvision_models, mb_sam
from common import IMAGENET_NORM, IMAGENET_MIN

MODULES = [m2f, sam, unet, torchvision_models, segmentation_pytorch, mb_sam]


def all_models():
    for module in MODULES:
        for model in module.models.keys():
            yield (module, model)


def get_patches(input_shape, target_shape):
    """Get non overlapping patches in (top, bottom, height, width) format"""
    (m1, n1), (m2, n2) = input_shape, target_shape
    if m1 <= m2 and n1 <= n2:
        yield 0, 0, *(x + 32 - (x % 32) for x in input_shape)
    else:
        for i in range(0, n1, n2):
            for j in range(0, m1, m2):
                #yield i, j, min(n2, n1-i), min(m2, m1-j)
                yield i, j, n2, m2


def train(
    model, train_loader, val_loader, checkpoint_dir, epochs,
    lr=1e-3, eval_frequency=80, warmup_steps=200, use_amp=False,
    clip_gradients=False
):
    scaler = GradScaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr*1e-2)
    lr_log = log10(lr)
    best_score = 0.
    last_update_step = 0
    epoch = 1
    global_step = 1
    patience = 20*eval_frequency
    model.train()

    def make_timer():
        begin = time.time()

        def tick(i):
            return (time.time()-begin)/i
        return tick

    tick = make_timer()
    while True:
        if epoch > epochs:
            print()
            return
        train_loss = 0
        for step, (imgs, msks) in enumerate(train_loader, 1):
            imgs = imgs.to("cuda", non_blocking=True)
            msks = msks.to("cuda", non_blocking=True)
            with autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(imgs)
                # print(imgs.size(), pred.size())
                loss = dice_loss(pred, msks.float()) \
                    + F.binary_cross_entropy_with_logits(pred, msks.float())
            if loss.isnan():
                print("  nan loss!!!")
                continue
            train_loss += loss.item()
            print(
                f"\r{epoch:3d} {global_step:6d} {train_loss/step:.03f} "
                f"{loss.cpu().item():.3f}", end=""
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip_gradients:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
            scaler.step(optimizer)
            scaler.update()

            if global_step % eval_frequency == 0:
                eval_loss = 0
                iou = 0
                fscore = 0
                model.eval()
                for i, (imgs, msks) in enumerate(val_loader):
                    imgs = imgs.to("cuda", non_blocking=True)
                    msks = msks.to("cuda", non_blocking=True)
                    with autocast(
                        "cuda", dtype=torch.float16, enabled=use_amp
                    ):
                        with torch.inference_mode(), torch.no_grad():
                            pred = model(imgs)
                            loss = dice_loss(pred, msks.float()) \
                                + F.binary_cross_entropy_with_logits(
                                    pred, msks.float()
                                )

                    eval_loss += loss.item()
                    iou += get_iou(pred, msks).item()
                    f = f1_score(pred, msks)
                    if not torch.isnan(f):
                        fscore += f.item()

                eval_loss /= (i + 1)
                iou /= (i + 1)
                fscore /= (i + 1)
                pe = tick(epoch)

                print(
                    f"\r{epoch:3d} {global_step:6d} {pe:3.1f}s "
                    f"train: {train_loss/step:.03f} val: {eval_loss:.03f} "
                    f"iou: {iou:.03f} f1: {fscore:.03f} ", end=""
                )
                if fscore > best_score:
                    best_score = fscore
                    last_update_step = global_step
                    print('\tBEST')
                    torch.save(
                        model.state_dict(),
                        checkpoint_dir + "/checkpoint_best.pth"
                    )
                    for row in range(imgs.size(0)):
                        TF.to_pil_image(
                            imgs[row]
                        ).save(f"{checkpoint_dir}/{row}-img.png")
                        TF.to_pil_image(
                            F.sigmoid(pred[row])
                        ).save(f"{checkpoint_dir}/{row}-pred.png")
                        TF.to_pil_image(
                            msks[row].float()
                        ).save(f"{checkpoint_dir}/{row}-true.png")
                    with open(f"{checkpoint_dir}/results", "w") as fp:
                        mem = torch.cuda.memory_reserved()/1024**3
                        print(
                            f"{epoch}\t{global_step}\t{pe}\t{mem}\t{fscore}", file=fp
                        )
                elif (global_step - last_update_step) > patience:
                    print("\t done")
                    return
                else:
                    print()

                model.train()

            if global_step <= warmup_steps:
                lr = 10**(lr_log-2*(1-(global_step/warmup_steps)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"  LR is NOW {param_group['lr']:.03g}", end="    ")
            global_step += 1
        train_loss = 0
        epoch += 1


class PlainDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_fnames, mask_fnames, shape=(1024, 1024), augment=False
    ):
        self.normalize = v2.Normalize(**IMAGENET_NORM)

        if augment:
            self.augment = torch.jit.script(nn.Sequential(
                v2.RandomApply(torch.nn.ModuleList([
                    v2.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
                    )]), p=2/3
                ),
                v2.RandomApply(torch.nn.ModuleList([v2.Grayscale(3)]), p=5e-2),
                v2.RandomInvert(p=2.5e-2),
            ))
            self.augment_spatial = torch.jit.script(nn.Sequential(
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
            ))
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
        img = TF.convert_image_dtype(img, torch.float)

        mask = read_image(mask_fname, ImageReadMode.GRAY)
        mask = TF.crop(mask, *crop_indices)
        mask = (mask.to(torch.uint8) > 0).float()

        if self.augment is not None:
            img = self.augment(img)
            aug = self.augment_spatial(torch.cat([img, mask]))
            img, mask = aug[:3], aug[-1:]
        img = self.normalize(img)

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


def dice_loss(input, target, eps=torch.finfo(torch.float32).eps):
    intersection, union = intersection_union(F.sigmoid(input), target)
    return 1 - ((2 * intersection) / (union + eps))


def get_iou(input, target, eps=torch.finfo(torch.float32).eps):
    intersection, union = intersection_union(input >= 0, target)
    return intersection / (union + eps)


def f1_score(input, target):
    """Expects input as logits"""
    input = (input >= 0).view(-1)
    target = target.bool().view(-1)
    tp2 = 2*(input & target).sum()
    fp = (input & ~target).sum()
    fn = (~input & target).sum()
    return tp2 / (tp2+fp+fn)


if __name__ == "__main__":
    models = {f"{k.__name__}/{v}": (k, v) for k, v in all_models()}

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("model", choices=models)
    parser.add_argument("-C", "--clip-gradients", action="store_true")
    parser.add_argument("-P", "--pretrained", action="store_true")
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
    parser.add_argument("-w", "--warmup-steps", type=int, default=400)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_dataset, eval_dataset = read_data(args.data_path, args.shape)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=True, multiprocessing_context="fork"
    )
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        multiprocessing_context="fork"
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
            clip_gradients=args.clip_gradients
        )
    except KeyboardInterrupt:
        print()
        print("Received keyboard interrupt. bye!")

    mem = torch.cuda.memory_reserved()
    print(f"GPU: {mem/1024**3:.1f} GB reserved")

    sys.exit(0)
