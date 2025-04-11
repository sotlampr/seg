"""
Copyright (C) 2025  Sotiris Lamrpinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import sys

import torch
from torch import nn
from torchvision.transforms.v2.functional import resize

sys.path.insert(0, "../RootNav-2.0/training")
from rootnav2.hourglass import HourglassNet, Bottleneck  # noqa: E402
sys.path.remove("../RootNav-2.0/training")


models = {
    "hourglass": None
}


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
    assert not pretrained
    model = HourglassNet(Bottleneck, num_stacks=1, num_blocks=1, num_classes=1)
    return Model(model, optimize=optimize)
