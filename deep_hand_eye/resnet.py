from typing import Any, List, Optional, Union, Type

import torch
import torch.nn as nn

from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    _resnet,
)


class ConvOutResNet(ResNet):
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvOutResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    delattr(model, "avgpool")
    delattr(model, "fc")
    delattr(model, "maxpool")
    model.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    return model


def resnet18(
    *,
    weights: Optional[ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights The pretrained weights to use. Use None for no pretraining.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def resnet34(
    *,
    weights: Optional[ResNet34_Weights] = ResNet34_Weights.IMAGENET1K_V1,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """
    ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights The pretrained weights to use. Use None for no pretraining.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
