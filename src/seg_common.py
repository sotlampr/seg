"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
"""
import os

import torch

# needed before importing any modules as they refer to it
IMAGENET_NORM = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

IMAGENET_MIN = (
    0 - torch.tensor(IMAGENET_NORM["mean"])
    / torch.tensor(IMAGENET_NORM["std"])
)

DATA_ROOT = os.environ.get(
    "SEG_DATA_ROOT", os.path.join(os.getcwd(), os.pardir, "data"))

if not __package__:
    package_source_path = os.path.dirname(__file__)
else:
    package_source_path = None

import seg_m2f as m2f  # noqa: E402
import seg_mb_sam as mb_sam  # noqa: E402
import seg_rootnav as rootnav  # noqa: E402
import seg_sam as sam  # noqa: E402
import seg_sam2 as sam2  # noqa: E402
import seg_segmentation_pytorch as segmentation_pytorch  # noqa: E402
import seg_segroot as segroot  # noqa: E402
import seg_torchvision_models as torchvision_models  # noqa: E402
import seg_unet as unet  # noqa: E402
import seg_unet_valid as unet_valid  # noqa: E402

MODULES = [
    m2f, mb_sam, rootnav, sam, sam2, segmentation_pytorch, segroot,
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


def get_model_dict():
    return {f"{k.__name__.strip('seg_')}/{v}": (k, v) for k, v in all_models()}


def load_model(model_id, pretrained=False, optimize=True, models=None):
    if models is None:
        models = get_model_dict()
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
            "m2f", "sam", "sam2", "mb_sam", "unet", "unet_valid", "segroot",
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
