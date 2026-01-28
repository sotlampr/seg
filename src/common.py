"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
"""
import os

import torch

IMAGENET_NORM = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

IMAGENET_MIN = (
    0 - torch.tensor(IMAGENET_NORM["mean"])
    / torch.tensor(IMAGENET_NORM["std"])
)


import \
    m2f, mb_sam, rootnav, sam, samII, segmentation_pytorch, segroot, \
    torchvision_models, unet, unet_valid  # noqa: E401 E402


MODULES = [
    m2f, mb_sam, rootnav, sam, samII, segmentation_pytorch, segroot,
    torchvision_models, unet, unet_valid
]


def torch_init(seed=42, deterministic=False):
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)


def all_models():
    for module in MODULES:
        for model in module.models.keys():
            yield (module, model)


def load_model(model_id, pretrained=False, optimize=True, models=None):
    if models is None:
        models = {f"{k.__name__}/{v}": (k, v) for k, v in all_models()}
    module, model_name = models[model_id]
    return module.new(
        model_name, pretrained=pretrained, optimize=optimize
    )


def expand_filename(orig_fname, alternative_naming=False):
    """ Infer parameters from directory name.
    Expected format:
        $PACKAGE-$MODEL-$DATASET-$SEED(-pretrained)?
    """
    abs_fname = os.path.abspath(orig_fname)
    if os.path.isdir(abs_fname):
        _, parent_dir = os.path.split(orig_fname)
    else:
        _, parent_dir = os.path.split(os.path.split(orig_fname)[0])

    pkg, model, *attrs, last = parent_dir.split("-")
    if last == "pretrained":
        pretrained = True
        runid, dataset = attrs.pop(), attrs.pop()
    else:
        pretrained = False
        runid, dataset = last, attrs.pop()
    if alternative_naming:
        if pkg in {
            "m2f", "sam", "samII", "mb_sam", "unet", "unet_valid", "segroot",
            "rootnav"
        }:
            attrs = (model, *attrs)
            model = pkg
    return dict(
        orig_fname=orig_fname,
        package=pkg,
        model=model,
        dataset=dataset,
        runid=int(runid),
        pretrained=pretrained,
        attributes="-".join(attrs),
        model_variant="-".join((model, *attrs))
    )
