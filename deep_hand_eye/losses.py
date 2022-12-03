from typing import Optional, Tuple

import torch
import torch.nn as nn


class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = pred @ target.permute(0, 2, 1)
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
        rotation_loss: Optional[torch.nn.Module] = None,
        beta: float = 0,
        gamma: float = 0,
        learn_beta: bool = True,
        pose_type: str = "SE3",
    ):
        super().__init__()
        self.translation_loss = translation_loss
        if rotation_loss is None:
            if pose_type == "SE3":
                self.rotation_loss = GeodesicLoss()
            elif pose_type == "xyz_log_quaternion":
                self.rotation_loss = nn.L1Loss()
            else:
                raise ValueError
        else:
            self.rotation_loss = rotation_loss

        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)
        self.pose_type = pose_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pose_type == "SE3":
            trans_loss = self.translation_loss(pred[..., :3, -1], target[..., :3, -1])
            rot_loss = self.rotation_loss(pred[..., :3, :3], target[..., :3, :3])
        elif self.pose_type == "xyz_log_quaternion":
            trans_loss = self.translation_loss(pred[..., :3], target[..., :3])
            rot_loss = self.rotation_loss(pred[..., 3:], target[..., 3:])
        else:
            raise ValueError

        beta_exp = torch.exp(-self.beta)
        gamma_exp = torch.exp(-self.gamma)
        loss = beta_exp * trans_loss + self.beta + gamma_exp * rot_loss + self.gamma

        return loss, trans_loss, rot_loss
