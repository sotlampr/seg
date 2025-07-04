"""
Copyright (C) 2025  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import torch

IMAGENET_NORM = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

IMAGENET_MIN = (
    0 - torch.tensor(IMAGENET_NORM["mean"])
    / torch.tensor(IMAGENET_NORM["std"])
)
