import torch
from torch import nn

IMAGENET_NORM = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

IMAGENET_MIN = (
    0 - torch.tensor(IMAGENET_NORM["mean"])
    / torch.tensor(IMAGENET_NORM["std"])
)


class Upscaler(nn.Module):
    """Upscales mask to 4x, concatenates the upscaled the input,
    and runs an 1x1 convolutin to get the upscaled version.
    """
    def __init__(self):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(
            1, 3, kernel_size=12, stride=4, padding=4
        )
        self.combine = nn.Conv2d(
            6, 1, kernel_size=9, stride=1, padding=4
        )

    def forward(self, input, masks):
        upscaled = self.upscale(masks)
        return self.combine(torch.cat([input, upscaled], dim=1))
