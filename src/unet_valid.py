# pylint: disable=C0111, W0221, R0902
"""
U-Net architecture based on:
https://arxiv.org/abs/1505.04597
And modified to use Group Normalization
https://arxiv.org/abs/1803.08494
And then modified to use residual style connections.
With other alterations.

Copyright (C) 2019, 2020 Abraham George Smith
    Modifications by Sotiris Lamprinidis.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch import nn
from torchvision.transforms.v2.functional import center_crop

from utils import check_for_file, get_pretrained_fname


class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # need to keep track of output here for up phase.
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        half_channels = in_channels // 2

        # Now a 2x2 convolution that halves the feature channels
        # this also up-samples
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, half_channels,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, half_channels)
        )
        # 2 layers of 3x3 conv + relu
        # still uses full channels as half channels
        # is added from down side output
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, half_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, half_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(half_channels, half_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, half_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = center_crop(down_out, out.shape[2:])
        out = torch.cat([cropped, out], dim=1)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class UNetGN(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        # input image is 572 by 572
        # 3x3 relu conv with 64 kernels
        self.conv_in = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            # now at 570 x 570 due to valid padding.
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64)
            # now at 568 x 568, 64 channels
        )
        # need to keep track of output here for up phase.
        """ This down block makes the following transformations.
        to 284x284 with 64 channels
        to 282x282 with 128 channels
        to 280x280 with 128 channels
        """
        self.down1 = DownBlock(64)
        # need to keep track of output here for up phase.
        """ Makes following transformations
        to 140x140 with 128 channels
        to 138x138 with 256 channels
        to 136x136 with 256 channels
        """
        self.down2 = DownBlock(128)
        # need to keep track of output here for up phase

        """ Makes following transformations
        to 68x68 with 256 channels
        to 66x66 with 512 channels
        to 64x64 with 512 channels
        """
        self.down3 = DownBlock(256)
        # need to keep track of output here for up phase

        """ Makes following transformations
        to 32x32 with 512 channels
        to 30 x 30 with 1024 channels
        to 28x28 with 1024 channels
        """
        self.down4 = DownBlock(512)

        """ Makes following transformations
            to is 56x56 with 512 channels (up and conv)
            to is 56x56 with 1024 channels (concat)
            to is 54x54 with 512 channels (conv)
            to is 52x52 with 512 channels (conv)
        """
        self.up1 = UpBlock(1024)
        self.up2 = UpBlock(512)
        self.up3 = UpBlock(256)
        self.up4 = UpBlock(128)
        # output is now at 64x388x388
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        # output is now at 2x388x388
        # each layer in the output represents a class 'probability'
        # but these aren't really probabilities as the model is not
        # calibrated.

    def forward(self, x):
        out1 = self.conv_in(x)
        # (1, 64, 568, 568)
        out2 = self.down1(out1)
        # (1, 128, 280, 280)
        out3 = self.down2(out2)
        # (1, 256, 136, 136)
        out4 = self.down3(out3)
        # (1, 512, 64, 64)
        out = self.down4(out4)
        # (1, 1024, 28, 28)
        out = self.up1(out, out4)
        # (1, 512, 52, 52)
        out = self.up2(out, out3)
        # (1, 256, 100, 100)
        out = self.up3(out, out2)
        # (1, 128, 196, 196)
        out = self.up4(out, out1)
        # (1, 64, 388, 388)
        out = self.conv_out(out)
        # each layer in the output represents a class 'probability'
        # but these aren't really probabilities as the model is not
        # calibrated.
        return out


class DownBlockRes(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv1x1 = nn.Sequential(
            # down sample channels again.
            nn.Conv2d(in_channels*2, in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out1 = self.pool(x)
        out2 = self.conv1(out1)
        out3 = self.conv2(out2)
        out4 = self.conv1x1(out3)
        return out4 + out1


class UpBlockRes(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = center_crop(down_out, out.shape[2:])
        out = cropped + out # residual
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class UNetGNRes(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64)
            # now at 568 x 568, 64 channels
        )
        self.down1 = DownBlockRes(64)
        self.down2 = DownBlockRes(64)
        self.down3 = DownBlockRes(64)
        self.down4 = DownBlockRes(64)
        self.up1 = UpBlockRes(64)
        self.up2 = UpBlockRes(64)
        self.up3 = UpBlockRes(64)
        self.up4 = UpBlockRes(64)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        out1 = self.conv_in(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out = self.up1(out5, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = self.conv_out(out)
        return out


zenodo_base = "https://zenodo.org/records"


def get_url(zenodo_id, weights_fn):
    url = f"{zenodo_base}/{zenodo_id}/files/{weights_fn}?download=1"
    return url


models = {
    "GN": (UNetGN, 3484015, "checkpoint_73.pkl"),
    "GNRes": (UNetGNRes, None, None)
}


def new(name, pretrained=False, optimize=True):
    cls, zenodo_id, weights_fn = models[name]
    if pretrained:
        pretrained_weights = get_pretrained_fname(weights_fn)
        check_for_file(pretrained_weights, get_url, zenodo_id, weights_fn)
    else:
        pretrained_weights = None
    model = cls()
    if pretrained:
        state_dict = torch.load(pretrained_weights, weights_only=True)
        del state_dict["conv_out.0.weight"], state_dict["conv_out.0.bias"]
        del state_dict["conv_out.2.weight"], state_dict["conv_out.2.bias"]
        model.load_state_dict(state_dict, strict=False)
    return torch.compile(model) if optimize else model
