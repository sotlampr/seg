"""
Copyright (C) 2025  Sotiris Lamrpinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import os
import sys
import pickle

import torch
from torch import nn
from torchvision.transforms.v2.functional import resize
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from utils import check_for_file, get_pretrained_fname

sys.path.insert(0, "../Mask2Former")
import mask2former  # noqa: E402
from mask2former import MaskFormer, add_maskformer2_config  # noqa: E402
sys.path.remove("../Mask2Former")


upstream_url = \
    "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic"

models = {
    "R50": ("maskformer2_R50_bs16_50ep", "94dc52.pkl"),
    "R101": ("maskformer2_R101_bs16_50ep", "b807bd.pkl"),
    "swin-tiny": ("maskformer2_swin_tiny_bs16_50ep", "9fd0ae.pkl"),
    "swin-small": ("maskformer2_swin_small_bs16_50ep", "a407fd.pkl"),
    "swin-base": ("maskformer2_swin_base_384_bs16_50ep", "9d7f02.pkl"),
}

cfgs_path = os.path.join(
    os.path.split(mask2former.__file__)[0],
    "../configs/coco/panoptic-segmentation"
)


def get_url(model_name, weights_ext):
    url = f"{upstream_url}/{model_name}/model_final_{weights_ext}"
    return url


class Model(nn.Module):
    def __init__(self, model, optimize=True):
        super().__init__()
        self.model = torch.compile(model) if optimize else model

    def forward(self, x):
        features = self.model.backbone(x)
        outputs = self.model.sem_seg_head(features)
        masks = outputs["pred_masks"]
        return resize(masks, x.shape[2:])


def make_model(cfg_file, pretrained_weights=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_file)

    if pretrained_weights:
        cfg.MODEL.WEIGHTS = pretrained_weights

    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    model = MaskFormer(**MaskFormer.from_config(cfg))
    if pretrained_weights:
        with open(pretrained_weights, "rb") as fp:
            state_dict = {
                k: torch.tensor(v) for k, v in pickle.load(fp)["model"].items()
            }
        state_dict["sem_seg_head.predictor.query_feat.weight"] = \
            state_dict.pop("sem_seg_head.predictor.static_query.weight")
        del state_dict["sem_seg_head.predictor.class_embed.weight"]
        del state_dict["sem_seg_head.predictor.class_embed.bias"]
        del state_dict["sem_seg_head.predictor.query_feat.weight"]
        del state_dict["sem_seg_head.predictor.query_embed.weight"]
        del state_dict["criterion.empty_weight"]
        model.load_state_dict(state_dict, strict=False)
    return model


def new(model_name, pretrained=False, optimize=True):
    model_id, weights_ext = models[model_name]

    model_cfg = f"{model_id}.yaml"
    if model_name.startswith("swin"):
        model_cfg = os.path.join("swin", f"{model_id}.yaml")
    model_cfg = os.path.join(cfgs_path, model_cfg)
    if pretrained:
        pretrained_weights = get_pretrained_fname(f"model_final_{weights_ext}")
    else:
        pretrained_weights = None
    check_for_file(pretrained_weights, get_url, model_id, weights_ext)
    return Model(make_model(model_cfg, pretrained_weights), optimize=optimize)
