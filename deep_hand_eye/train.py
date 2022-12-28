import os
import random
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.utils.tensorboard as tb
from torch_geometric.loader import DataLoader

from deep_hand_eye.model import GCNet
from deep_hand_eye.utils import AttrDict
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.losses import PoseCriterion
from deep_hand_eye.eval.evaluate import eval
from deep_hand_eye.pose_utils import (
    qexp,
    quaternion_angular_error,
    rotation_matrix_to_quaternion,
    xyz_quaternion_to_xyz_log_quaternion
)


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.epochs = 45
config.save_dir = ""
config.model_name = ""
config.batch_size = 16
config.eval_freq = 2000
config.eval_samples = 2000  # Number of graphs to evaluate on
config.log_freq = 20  # Iters to log after
config.save_freq = 5  # Epochs to save after. Set None to not save.
config.rel_pose_coeff = 1  # Set None to remove this auxiliary loss
config.num_opt_iter_train = 1
config.num_opt_iter_test = 1
config.opt_start_epoch_train = 35
config.opt_start_epoch_test = 35
config.model_name = ""
config.log_dir = Path("runs")
config.model_save_dir = Path("models")


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self, config: AttrDict) -> None:
        self.config = config
        self.beta_loss_coeff = 0.0  # initial relative translation loss coeff
        self.gamma_loss_coeff = -3  # initial relative rotation loss coeff
        self.weight_decay = 0.0005
        self.lr = 5e-5
        self.lr_decay = 0.1  # learning rate decay factor
        self.lr_decay_step = 20
        self.learn_loss_coeffs = True
        self.run_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

        if self.config.device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.tb_writer = tb.writer.SummaryWriter(
            log_dir=self.config.log_dir / self.config.model_name / self.run_id
        )

        # Setup Dataset
        self.train_dataset = MVSDataset(
            image_folder="data/DTU_MVS_2014/Rectified/train/"
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        self.test_dataset = MVSDataset(image_folder="data/DTU_MVS_2014/Rectified/test/")
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        # Define the model
        self.model = GCNet(rel_pose=self.config.rel_pose_coeff is not None).to(
            self.device
        )
        self.model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        self.params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Define loss
        self.train_criterion_R = PoseCriterion(
            beta=self.beta_loss_coeff,
            gamma=self.gamma_loss_coeff,
            learn_beta=True,
            pose_type="xyz_log_quaternion",
        ).to(self.device)
        self.train_criterion_he = PoseCriterion(
            beta=self.beta_loss_coeff,
            gamma=self.gamma_loss_coeff,
            learn_beta=True,
            pose_type="SE3",
        ).to(self.device)

        # Define optimizer
        param_list = [{"params": self.model.parameters()}]
        if self.learn_loss_coeffs:
            param_list.append(
                {
                    "params": [
                        self.train_criterion_he.beta,
                        self.train_criterion_he.gamma,
                    ]
                }
            )

            if self.config.rel_pose_coeff is not None:
                param_list.append(
                    {
                        "params": [
                            self.train_criterion_R.beta,
                            self.train_criterion_R.gamma,
                        ]
                    }
                )

        self.optimizer = torch.optim.Adam(
            params=param_list,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def train(self):
        iter_no = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            if epoch > 1 and epoch % self.lr_decay_step == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * self.lr_decay
                    print("LR: ", param_group["lr"])

            for _, data in tqdm(
                enumerate(self.train_dataloader),
                desc=f"[Epoch {epoch:04d}/{self.config.epochs}] train",
                total=len(self.train_dataloader),
            ):
                self.optimizer.zero_grad()

                # Get model predictions
                data = data.to(self.device)
                target_he, target_R = data.y, data.y_edge
                pred_he, pred_R, _ = self.model(
                    data=data,
                    opt_iterations=0 if epoch < config.opt_start_epoch_train else config.num_opt_iter_train
                )

                # Calculate losses
                loss_he, loss_he_t, loss_he_q = self.train_criterion_he(
                    pred=pred_he,
                    target=target_he
                )
                loss_total = loss_he

                if self.config.rel_pose_coeff is not None:
                    loss_R, loss_R_t, loss_R_q = self.train_criterion_R(
                        pred=pred_R,
                        target=xyz_quaternion_to_xyz_log_quaternion(target_R)
                    )
                    loss_total += self.config.rel_pose_coeff * loss_R

                loss_total.backward()
                self.optimizer.step()

                if iter_no > 0 and iter_no % self.config.log_freq == 0:
                    self.tb_writer.add_scalar("train/epoch", epoch, iter_no)
                    self.tb_writer.add_scalar("train/total_loss", loss_total, iter_no)
                    self.tb_writer.add_scalar("train/he_pose_loss", loss_he, iter_no)
                    self.tb_writer.add_scalar("train/he_trans_loss", loss_he_t, iter_no)
                    self.tb_writer.add_scalar("train/he_rot_loss", loss_he_q, iter_no)
                    self.tb_writer.add_scalar(
                        "train/he_beta", self.train_criterion_he.beta, iter_no
                    )
                    self.tb_writer.add_scalar(
                        "train/he_gamma", self.train_criterion_he.gamma, iter_no
                    )
                    if self.config.rel_pose_coeff is not None:
                        self.tb_writer.add_scalar(
                            "train/rel_pose_loss", loss_R, iter_no
                        )
                        self.tb_writer.add_scalar(
                            "train/rel_trans_loss", loss_R_t, iter_no
                        )
                        self.tb_writer.add_scalar(
                            "train/rel_rot_loss", loss_R_q, iter_no
                        )
                        self.tb_writer.add_scalar(
                            "train/rel_beta", self.train_criterion_R.beta, iter_no
                        )
                        self.tb_writer.add_scalar(
                            "train/rel_gamma", self.train_criterion_R.gamma, iter_no
                        )

                if iter_no % self.config.eval_freq == 0:
                    _ = eval(
                        model=self.model,
                        dataloader=self.test_dataloader,
                        device=self.device,
                        num_samples=self.config.eval_samples,
                        eval_rel_pose=self.config.rel_pose_coeff is not None,
                        opt_iterations=0 if epoch < config.opt_start_epoch_test else config.num_opt_iter_test,
                        iter_no=iter_no,
                        tb_writer=self.tb_writer

                    )
                    self.model.train()
                iter_no += 1

            if (
                epoch > 0
                and self.config.save_freq is not None
                and (epoch + 1) % self.config.save_freq == 0
            ):
                self.save_model()

    def save_model(self):
        save_dir = self.config.model_save_dir / self.config.model_name / self.run_id
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model, save_dir / "model.pt")


def main():
    seed_everything(0)
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
