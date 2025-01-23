import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp


models = {
    "unet++-mobilenet_v2": smp.UnetPlusPlus,
    "unet++-resnet18": smp.UnetPlusPlus,
    "unet++-resnet50": smp.UnetPlusPlus,
    "unet++-resnet101": smp.UnetPlusPlus,
    "unet++-timm-efficientnet-b1": smp.UnetPlusPlus,
    "unet++-timm-efficientnet-b2": smp.UnetPlusPlus,
    "unet++-timm-efficientnet-b4": smp.UnetPlusPlus,
    "unet++-timm-efficientnet-b8": smp.UnetPlusPlus,
    "deeplabv3+-resnet50": smp.DeepLabV3Plus,
    "deeplabv3+-resnet101": smp.DeepLabV3Plus,
    "manet-resnet50": smp.MAnet,
    "manet-resnet101": smp.MAnet,
    "pan-resnet50": smp.PAN,
    "pan-resnet101": smp.PAN,
    "linknet-mobilenet_v2": smp.Linknet,
    "linknet-timm-efficientnet-b1": smp.Linknet,
    "linknet-timm-efficientnet-b2": smp.Linknet,
    "linknet-timm-efficientnet-b4": smp.Linknet,
    "linknet-timm-efficientnet-b8": smp.Linknet,
    "linknet-resnet50": smp.Linknet,
    "linknet-resnet101": smp.Linknet,
    "pspnet-resnet50": smp.PSPNet,
    "pspnet-resnet101": smp.PSPNet,
    "segformer-resnet18": smp.Segformer,
    "segformer-resnet50": smp.Segformer,
    "segformer-resnet101": smp.Segformer
}


class ModelShim(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, x.shape[2:])
        return out


def new(name, pretrained=False):
    model_name, encoder_name = name.split("-", maxsplit=1)
    model_cls = models[name]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None
    )
