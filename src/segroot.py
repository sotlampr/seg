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

from utils import check_for_file, get_pretrained_fname

sys.path.insert(0, "../SegRoot/code")
from model import SegRoot  # noqa: E402
sys.path.remove("../SegRoot/code")

models = {
    "w8d5": {
        "width": 8, "depth": 5, "weights": "best_segnet-(8,5)-0.6441.pt"
    },
    "w16d4": {"width": 16, "depth": 4, "weights": None},
    "w32d5": {"width": 32, "depth": 5, "weights": None},
    "w64d4": {"width": 64, "depth": 4, "weights": None},
    "w64d5": {"width": 64, "depth": 4, "weights": None}
}


def get_url(_, __, weights_fn):
    return f"https://github.com/wtwtwt0330/SegRoot/raw/refs/heads/master/weights/{weights_fn}"  # noqa: E501


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

    model_kwargs = models[model_name].copy()

    # 'weights' is only for pretrained init; SegRoot must not see it
    weights_id = model_kwargs.pop("weights", None)
    model = Model(SegRoot(**model_kwargs, num_classes=1))

    if pretrained:
        assert weights_id is not None
        weights_fn = get_pretrained_fname(weights_id)
        check_for_file(weights_fn, get_url, None, None, weights_id)
        model.load_state_dict(torch.load(weights_fn, weights_only=False))

    return torch.compile(model) if optimize else model
