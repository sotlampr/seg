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

sys.path.insert(0, "../SegRoot/code")
from model import SegRoot  # noqa: E402
sys.path.remove("../SegRoot/code")

models = {
    "w16d4": {"width": 16, "depth": 4},
    "w32d5": {"width": 32, "depth": 5},
    "w64d4": {"width": 64, "depth": 4},
    "w64d5": {"width": 64, "depth": 4}
}


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoders = model.encoders
        self.decoders = model.decoders
        self.classifier = model.classifier

    def forward(self, x):
        # custom forward as the original model does not return logits
        indices = []
        for enc in self.encoders:
            x, ind = enc(x)
            indices.append(ind)
        for dec, ind in zip(self.decoders, indices[::-1]):
            x = dec(x, ind)
        return self.classifier(x)


def new(model_name, pretrained=False, optimize=True):
    assert not pretrained

    model_kwargs = models[model_name]
    model = Model(SegRoot(**model_kwargs, num_classes=1))
    return torch.compile(model) if optimize else model
