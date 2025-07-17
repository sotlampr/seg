"""
Copyright (C) 2025  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import sys

import torch
from torch import nn
from torchvision.transforms.v2.functional import resize

from utils import check_for_file, get_pretrained_fname

sys.path.insert(0, "../RootNav-2.0/training")
from rootnav2.hourglass import HourglassNet, Bottleneck  # noqa: E402
sys.path.remove("../RootNav-2.0/training")

upstream_url = \
    "https://cvl.cs.nott.ac.uk/resources/trainedmodels/"

models = {
    "hourglass": "wheat_bluepaper-6d109612.pth"
}


def get_url(model_name, weights_fn):
    url = f"{upstream_url}/{weights_fn}"
    return url


class Model(nn.Module):
    def __init__(self, model, upscale=False, optimize=True):
        super().__init__()
        self.model = torch.compile(model) if optimize else model

        # double the kernel and stride so we get the same
        # resolution as output
        self.model.tail_deconv2 = nn.ConvTranspose2d(
            128, 128, kernel_size=8, stride=4, padding=2
        )

    def forward(self, x):
        masks, *_ = self.model(x)
        return resize(masks, x.shape[2:])


def new(model_name, pretrained=False, optimize=True):
    weights_fn = models[model_name]
    if pretrained:
        pretrained_weights = get_pretrained_fname(weights_fn)
    else:
        pretrained_weights = None
    check_for_file(pretrained_weights, get_url, weights_fn)
    model = HourglassNet(Bottleneck, num_stacks=1, num_blocks=1, num_classes=1)
    if pretrained:
        state_dict = \
            torch.load(pretrained_weights, weights_only=False)["model_state"]
        state_dict = {
            k.split(".", maxsplit=1)[-1]: v for k, v in state_dict.items()}
        del state_dict["score.0.weight"], state_dict["score.0.bias"]
        model.load_state_dict(state_dict, strict=False)
    return Model(model, optimize=optimize)
