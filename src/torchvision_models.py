import torch
import torch.nn.functional as F
import torchvision.models.segmentation as seg_models


models = {
    "deeplabv3-mobilenet_v3_large":
        seg_models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
    "deeplabv3-resnet50":
        seg_models.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    "deeplabv3-resnet101":
        seg_models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    "fcn-resnet50":
        seg_models.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    "fcn-resnet101":
        seg_models.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    "lraspp-mobilenet_v3_large":
        seg_models.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
}


class Model(torch.nn.Module):
    """Replace the classifier head with a fresh one with 2 classes, get the
    "out" key in forward, crop to 388x388.
    """
    def __init__(self, model):
        super().__init__()
        self.model = torch.compile(model)
        clfcls = model.classifier.__class__
        if clfcls.__name__ == 'LRASPPHead':
            self.model.classifier = clfcls(40, 960, 1, 128)
        else:
            in_channels = next(model.classifier.parameters()).size(1)
            self.model.classifier = clfcls(in_channels, 1)

    def forward(self, x):
        out = self.model(x)["out"]
        # print(x.shape, out.shape)
        # import ipdb; ipdb.set_trace()
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, x.shape[2:])
        return out


def new(name, pretrained=False):
    model_name, encoder_name = name.split("-")
    if pretrained:
        weights = models[name]
    else:
        weights = None

    model_cls = getattr(seg_models, model_name + "_" + encoder_name)
    return Model(model_cls(weights=weights))
