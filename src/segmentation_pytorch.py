import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

base_models = {
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "linknet": smp.Linknet,
    "manet": smp.MAnet,
    "pan": smp.PAN,
    "pspnet": smp.PSPNet,
    "segformer": smp.Segformer,
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
}


encoders = (
    "densenet121",
    "densenet161",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "timm-efficientnet-b5",
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mobilenet_v2",
    "mobileone_s0",
    "mobileone_s1",
    "resnet18",
    "resnet50",
    "vgg11",
    "vgg19",
    "xception",
    "inceptionv4",
)

models = {
    f"{name}-{enc}": cls
    for name, cls in base_models.items()
    for enc in encoders
}


class ModelShim(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = torch.jit.script(model)

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
