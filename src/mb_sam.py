"""
Copyright (C) 2025, 2026  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.
"""
import torch
import warnings

from torch import nn
from torchvision.transforms.v2.functional import pad, center_crop, resize

with warnings.catch_warnings(action="ignore"):
    from mobile_sam import build_sam_vit_t

from common import IMAGENET_MIN
from utils import check_for_file, get_pretrained_fname


models = {
    "vit_t": ("mobile_sam.pt",)
}

url = "https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view"


def get_url(model_id, weights_ext=None):
    return url


class Model(nn.Module):
    def __init__(self, model, optimize=True):
        super().__init__()
        self.model = torch.compile(model) if optimize else model

    def forward(self, x):
        # Sam always needs 1024x1024 inputs
        pad_h, pad_w = ((1024-n)//2 for n in x.shape[2:])
        if pad_h or pad_w:
            img_embed = self.model.image_encoder(
                pad(x, (pad_w, pad_h), fill=IMAGENET_MIN)
            )
        else:
            img_embed = self.model.image_encoder(x)

        prompt_embeds = self.model.prompt_encoder(None, None, None)
        masks, _ = self.model.mask_decoder(
            img_embed, self.model.prompt_encoder.get_dense_pe(),
            *prompt_embeds, False
        )
        return resize(
            center_crop(masks, (n//4 for n in x.shape[2:]))
            if pad_h or pad_w else masks,
            x.shape[2:]
        )


def new(model_name, pretrained=False, optimize=True):
    weights_fname, *_ = models[model_name]
    if pretrained:
        checkpoint = get_pretrained_fname(weights_fname)
    else:
        checkpoint = None
    check_for_file(checkpoint, get_url, "mb_sam", weights_fname)
    return Model(build_sam_vit_t(checkpoint=checkpoint), optimize=optimize)
