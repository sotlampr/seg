from torch import nn
import torch.nn.functional as F

import segment_anything

from utils import check_for_file, get_pretrained_fname  # noqa: E402


upstream_url = \
    "https://dl.fbaipublicfiles.com/segment_anything"

models = {
    "vit-base": ("vit_b", "01ec64.pth",),
    "vit-large": ("vit_l", "0b3195.pth",),
    # Too big (vit-large is already around 24G)
    # "vit-huge": ("vit_h", "4b8939.pth",)
}


def get_url(model_id, weights_ext):
    return f"{upstream_url}/sam_{model_id}_{weights_ext}"


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        img_embed = self.model.image_encoder(x)
        prompt_embeds = self.model.prompt_encoder(None, None, None)
        masks, _ = self.model.mask_decoder(
            img_embed, self.model.prompt_encoder.get_dense_pe(),
            *prompt_embeds, False
        )
        return F.interpolate(masks, x.shape[2:])


def new(model_name, pretrained=False):
    model_id, weights_ext = models[model_name]
    fn = getattr(segment_anything, f"build_sam_{model_id}")
    if pretrained:
        checkpoint = get_pretrained_fname(f"sam_{model_id}_{weights_ext}")
    else:
        checkpoint = None
    check_for_file(checkpoint, get_url, model_id, weights_ext)
    return Model(fn(checkpoint=checkpoint))
