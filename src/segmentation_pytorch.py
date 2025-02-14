"""
Copyright (C) 2025  Sotiris Lamrpinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# (model_class, non_dilated, supports_mit)
base_models = {
    "deeplabv3": (smp.DeepLabV3, False, True),
    "deeplabv3+": (smp.DeepLabV3Plus, False, True),
    "pan": (smp.PAN, False, True),
    "manet": (smp.MAnet, True, True),
    "pspnet": (smp.PSPNet, True, True),
    "segformer": (smp.Segformer, True, True),
    "unet": (smp.Unet, True, True),
    "linknet": (smp.Linknet, True, False),
    "unet++": (smp.UnetPlusPlus, True, False),
}


dilated_encoders = (
    "efficientnet-b0",  # 4M
    "efficientnet-b1",  # 6M
    "efficientnet-b2",  # 7M
    "efficientnet-b3",  # 10M
    "efficientnet-b4",  # 17M
    "efficientnet-b5",  # 28M
    "efficientnet-b6",  # 40M
    "timm-efficientnet-b0",  # 4M
    "timm-efficientnet-b1",  # 6M
    "timm-efficientnet-b2",  # 7M
    "timm-efficientnet-b3",  # 10M
    "timm-efficientnet-b4",  # 17M
    "timm-efficientnet-b5",  # 28M
    "timm-efficientnet-b6",  # 40M
    "mobilenet_v2",  # 2M
    "mobileone_s0",  # 4M
    "mobileone_s1",  # 3M
    "mobileone_s2",  # 5M
    "mobileone_s3",  # 8M
    "mobileone_s4",  # 12M
    "resnet18",  # 11M
    "resnet50",  # 23M
    "resnet101",  # 42M
)

non_dilated_encoders = (
    "densenet121",  # 6M
    "densenet169",  # 12M
    "densenet201",  # 18M
    "densenet161",  # 26M
    "vgg11",  # 9M
    "vgg19",  # 20M
    "xception",  # 20M
    "inceptionv4",  # 41M
)

mit_encoders = (
    "mit_b0",  # 3M
    "mit_b1",  # 13M
    "mit_b2",  # 24M
    "mit_b3",  # 44M
)

models = {
    **{
        f"{name}-{enc}": cls
        for name, (cls, _, _) in base_models.items()
        for enc in dilated_encoders
    },
    **{
        f"{name}-{enc}": cls
        for name, (cls, non_dilated, _) in base_models.items()
        for enc in non_dilated_encoders
        if non_dilated
    },
    **{
        f"{name}-{enc}": cls
        for name, (cls, _, supports_mit) in base_models.items()
        for enc in mit_encoders
        if supports_mit
    }
}


def new(name, pretrained=False, optimize=True):
    model_name, encoder_name = name.split("-", maxsplit=1)
    model_cls = models[name]
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None
    )
    return torch.compile(model) if optimize else model
