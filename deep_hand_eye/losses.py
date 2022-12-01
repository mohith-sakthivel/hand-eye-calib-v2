from typing import Tuple

import torch
import torch.nn as nn


class GeodesicLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = pred @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        trace = delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        distance = ((trace - 1) / 2).clamp(-1 + self.eps, 1 - self.eps)
        distance = torch.arccos(distance)
        if self.reduction == "none":
            return distance
        elif self.reduction == "mean":
            return distance.mean()
        elif self.reduction == "sum":
            return distance.sum()


class PoseCriterion(nn.Module):
    def __init__(
        self,
        translation_loss: torch.nn.Module = nn.L1Loss(),
        quaternion_loss: torch.nn.Module = nn.L1Loss(),
        beta: float = 0,
        gamma: float = 0,
        learn_beta: bool = True,
    ):
        super().__init__()
        self.translation_loss = translation_loss
        self.quaternion_loss = quaternion_loss
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param pred: N x 6
        :param targ: N x 6
        """
        t_loss = self.translation_loss(pred[..., :3], targ[..., :3])
        q_loss = self.quaternion_loss(pred[..., 3:], targ[..., 3:])
        beta_exp = torch.exp(-self.beta)
        gamma_exp = torch.exp(-self.gamma)
        loss = beta_exp * t_loss + self.beta + gamma_exp * q_loss + self.gamma

        return loss, t_loss, q_loss


class PoseCriterionSE3(nn.Module):
    def __init__(
        self,
        translation_loss: torch.nn.Module = nn.L1Loss(),
        rotation_loss: torch.nn.Module = GeodesicLoss(),
        beta: float = 0,
        gamma: float = 0,
        learn_beta: bool = True,
    ):
        super().__init__()
        self.translation_loss = translation_loss
        self.rotation_loss = rotation_loss
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        t_loss = self.translation_loss(pred[..., :3, -1], targ[..., :3, -1])
        q_loss = self.rotation_loss(pred[..., :3, :3], targ[..., :3, :3])
        beta_exp = torch.exp(-self.beta)
        gamma_exp = torch.exp(-self.gamma)
        loss = beta_exp * t_loss + self.beta + gamma_exp * q_loss + self.gamma

        return loss, t_loss, q_loss
