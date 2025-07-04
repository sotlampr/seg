"""
Copyright (C) 2025  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import torch
from torch import nn
from torchvision.transforms.v2.functional import pad, center_crop, resize

from sam2.build_sam import build_sam2

from common import IMAGENET_MIN
from utils import check_for_file, get_pretrained_fname
upstream_url = "https://dl.fbaipublicfiles.com/segment_anything_2"


models = {
    "hiera-tiny": ("092824", "sam2.1_hiera_tiny.pt", "t"),
    "hiera-small": ("092824", "sam2.1_hiera_small.pt", "s"),
    "hiera-base_plus": ("092824", "sam2.1_hiera_base_plus.pt", "b+"),
    "hiera-large": ("092824", "sam2.1_hiera_large.pt", "l"),
}


def get_url(rev, chkpt, cid):
    return f"{upstream_url}/{rev}/{chkpt}"


class Model(nn.Module):
    def __init__(self, model, optimize=True):
        super().__init__()
        self.model = torch.compile(model) if optimize else model
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def forward(self, x):
        # Sam always needs 1024x1024 inputs
        pad_h, pad_w = ((1024-n)//2 for n in x.shape[2:])
        if pad_h or pad_w:
            bb_out = self.model.forward_image(
                pad(x, (pad_w, pad_h), fill=IMAGENET_MIN)
            )
        else:
            bb_out = self.model.forward_image(x)

        _, vision_feats, _, _ = self.model._prepare_backbone_features(bb_out)
        feats = [
            feat.permute(1, 2, 0).view(x.size(0), -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self._bb_feat_sizes[::-1]
            )
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        zero_prompt = True
        if zero_prompt:
            # just provide (0, 0) as the point of query
            points = torch.zeros((x.size(0), 1, 2), device=x.device)
            # label 1 is supposed to be fg
            labels = torch.ones((x.size(0), 1), device=x.device)
            sparse_emb, dense_emb = self.model.sam_prompt_encoder(
                points=(points, labels), boxes=None, masks=None,
            )

        else:
            sparse_emb, dense_emb = self.model.sam_prompt_encoder(
                points=None, boxes=None, masks=None,
            )

        masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=features["high_res_feats"],
        )

        return resize(
            center_crop(masks, (n//4 for n in x.shape[2:]))
            if pad_h or pad_w else masks,
            x.shape[2:]
        )


def new(model_name, pretrained=False, optimize=True):
    rev, chkpt, cid = models[model_name]
    if pretrained:
        chkpt = get_pretrained_fname(chkpt.split("/")[-1])
        check_for_file(chkpt, get_url, rev, chkpt, cid)
    model = build_sam2(
        f"configs/sam2.1/sam2.1_hiera_{cid}.yaml",
        ckpt_path=get_pretrained_fname(chkpt) if pretrained else None
    )
    return Model(model, optimize=optimize)
