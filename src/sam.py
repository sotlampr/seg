import torch
from torch import nn
from torchvision.transforms.v2.functional import pad, center_crop

import segment_anything

from common import Upscaler, IMAGENET_MIN
from utils import check_for_file, get_pretrained_fname


upstream_url = \
    "https://dl.fbaipublicfiles.com/segment_anything"

models = {
    "vit-base": ("vit_b", "01ec64.pth",),
    "vit-large": ("vit_l", "0b3195.pth",),
    "vit-huge": ("vit_h", "4b8939.pth",)
}


def get_url(model_id, weights_ext):
    return f"{upstream_url}/sam_{model_id}_{weights_ext}"


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = torch.compile(model)
        self.upscale = torch.compile(Upscaler())

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
        return self.upscale(x, center_crop(masks, (n//4 for n in x.shape[2:])))


def new(model_name, pretrained=False):
    model_id, weights_ext = models[model_name]
    fn = getattr(segment_anything, f"build_sam_{model_id}")
    if pretrained:
        checkpoint = get_pretrained_fname(f"sam_{model_id}_{weights_ext}")
    else:
        checkpoint = None
    check_for_file(checkpoint, get_url, model_id, weights_ext)
    return Model(fn(checkpoint=checkpoint))
