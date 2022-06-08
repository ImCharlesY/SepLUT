import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class LightBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to use an extra pooling layer at the end
            of the backbone. Default: False.
        n_base_feats (int, optional): Channel multiplier. Default: 8.
    """

    def __init__(self,
                 input_resolution=256,
                 extra_pooling=False,
                 n_base_feats=8,
                 **kwargs) -> None:
        body = [BasicBlock(3, n_base_feats, stride=2, norm=True)]
        n_feats = n_base_feats
        for _ in range(3):
            body.append(
                BasicBlock(n_feats, n_feats * 2, stride=2, norm=True))
            n_feats = n_feats * 2
        body.append(BasicBlock(n_feats, n_feats, stride=2))
        body.append(nn.Dropout(p=0.5))
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = n_feats * (
            4 if extra_pooling else (input_resolution // 32) ** 2)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class Res18Backbone(nn.Module):
    r"""The ResNet-18 backbone.

    Args:
        pretrained (bool, optional): Whether to use the torchvison pretrained weights.
            Default: True.
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 224.
    """

    def __init__(self, pretrained=True, input_resolution=224, **kwargs):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)