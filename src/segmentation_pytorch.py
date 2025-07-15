"""
Copyright (C) 2025  Sotiris Lamprinidis

This program is free software and all terms of the GNU General Public License
version 3 as published by the Free Software Foundation apply. See the LICENSE
file in the root directory of the project or <https://www.gnu.org/licenses/>
for more details.

"""
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# (model_class, non_dilated)
base_models = {
    "deeplabv3": (smp.DeepLabV3, False),
    "deeplabv3+": (smp.DeepLabV3Plus, False),
    "pan": (smp.PAN, True),
    "manet": (smp.MAnet, True),
    "pspnet": (smp.PSPNet, True),
    "unet": (smp.Unet, True),
    "linknet": (smp.Linknet, True),
    "unet++": (smp.UnetPlusPlus, True),
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


segformer_models = (
    ("mit_b0", "smp-hub/segformer-b2-1024x1024-city-160k"),  # 3M
    ("mit_b1", "smp-hub/segformer-b2-1024x1024-city-160k"),  # 13M
    ("mit_b2", "smp-hub/segformer-b2-1024x1024-city-160k"),  # 24M
    ("mit_b3", "smp-hub/segformer-b3-1024x1024-city-160k")  # 44M
)

models = {
    **{
        f"{name}-{enc}": cls
        for name, (cls, _) in base_models.items()
        for enc in dilated_encoders
    },
    **{
        f"{name}-{enc}": cls
        for name, (cls, non_dilated) in base_models.items()
        for enc in non_dilated_encoders if non_dilated
    },
    **{
        f"segformer-{enc}": value
        for enc, value in segformer_models

    }
}


default_kwargs = {
    "in_channels": 3, "classes": 1, "activation": None
}


def new(name, pretrained=False, optimize=True):
    model_name, encoder_name = name.split("-", maxsplit=1)
    if model_name == "segformer":
        if pretrained:
            pt_name = models[name]
            model = smp.Segformer.from_pretrained(pt_name)
            model.segmentation_head[0] = torch.nn.Conv2d(
                model.segmentation_head[0].in_channels, 1, 1
            )
            model.classes = 1
        else:
            model = smp.Segformer(encoder_name, **default_kwargs)
    else:
        model_cls = models[name]
        model = model_cls(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            **default_kwargs
        )
        if optimize:
            model = torch.compile(model)
    return model
