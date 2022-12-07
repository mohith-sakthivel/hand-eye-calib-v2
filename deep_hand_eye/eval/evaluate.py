import os
import random
import numpy as np
from tqdm import tqdm
from typing import Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader

from deep_hand_eye.utils import AttrDict
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.pose_utils import (
    qexp,
    quaternion_angular_error,
    rotation_matrix_to_quaternion,
)


MODEL_PATH = "models/12_add_diff_opt/model.pt"
DATA_PATH = "data/DTU_MVS_2014/Rectified/test/"
NUM_SAMPLES = 2000


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.batch_size = 2


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def translation_criterion(pred, target):
    return np.linalg.norm(pred - target)


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_samples: int = 2000,
    eval_rel_pose: bool = True,
    opt_iterations: int = 0,
    iter_no: Optional[int] = None,
    tb_writer: Optional[SummaryWriter] = None,
):
    model.eval()

    rotation_criterion = quaternion_angular_error
    t_loss_he = []
    q_loss_he = []
    if eval_rel_pose:
        t_loss_R = []
        q_loss_R = []

    curr_samples = 0

    # inference loop
    for _, data in tqdm(
        enumerate(dataloader),
        desc="Eval" if iter_no is None else f"[Iter {iter_no:04d}] eval",
        total=num_samples / dataloader.batch_size,
    ):

        curr_samples += data.num_graphs
        data = data.to(device)
        output_he, output_R, _ = model(data, opt_iterations=opt_iterations)
        output_he = output_he.cpu().data.numpy()
        target_he = data.y.to("cpu").numpy()

        # normalize the predicted quaternions
        q = [rotation_matrix_to_quaternion(p[:3, :3]) for p in output_he]
        output_he = np.hstack((output_he[:, :3, 3], np.asarray(q)))
        q = [rotation_matrix_to_quaternion(p[:3, :3]) for p in target_he]
        target_he = np.hstack((target_he[:, :3, 3], np.asarray(q)))

        # calculate losses
        for p, t in zip(output_he, target_he):
            t_loss_he.append(translation_criterion(p[:3], t[:3]))
            q_loss_he.append(rotation_criterion(p[3:], t[3:]))

        if eval_rel_pose:
            output_R = output_R.cpu().data.numpy()
            # normalize the predicted quaternions
            target_R = data.y_edge.to("cpu").numpy()

            q = [qexp(p[3:]) for p in output_R]
            output_R = np.hstack((output_R[:, :3], np.asarray(q)))

            for p, t in zip(output_R, target_R):
                t_loss_R.append(translation_criterion(p[:3], t[:3]))
                q_loss_R.append(rotation_criterion(p[3:], t[3:]))

        if curr_samples > num_samples:
            break

    metrics = {}
    metrics["median_t_he"] = np.median(t_loss_he)
    metrics["median_q_he"] = np.median(q_loss_he)
    metrics["mean_t_he"] = np.mean(t_loss_he)
    metrics["mean_q_he"] = np.mean(q_loss_he)
    metrics["max_t_he"] = np.max(t_loss_he)
    metrics["max_q_he"] = np.max(q_loss_he)

    if iter_no is not None:
        print("Iter No [{iter_no:04d}]")

    print(
        "Error in translation: \n"
        "\t median {0:3.4f} m \n"
        "\t mean {1:3.4f} m \n"
        "\t max {2:3.4f} m \n".format(
            metrics["median_t_he"], metrics["mean_t_he"], metrics["max_t_he"]
        )
    )

    print(
        "Error in rotation: \n"
        "\t median {0:3.2f} degrees \n"
        "\t mean {1:3.2f} degrees \n"
        "\t max {2:3.2f} degrees \n".format(
            metrics["median_q_he"], metrics["mean_q_he"], metrics["max_q_he"]
        )
    )

    if tb_writer is not None:
        tb_writer.add_scalar("test/he_trans_median", metrics["median_t_he"], iter_no)
        tb_writer.add_scalar("test/he_trans_mean", metrics["mean_t_he"], iter_no)
        tb_writer.add_scalar("test/he_trans_max", metrics["max_t_he"], iter_no)
        tb_writer.add_scalar("test/he_rot_median", metrics["median_q_he"], iter_no)
        tb_writer.add_scalar("test/he_rot_mean", metrics["mean_q_he"], iter_no)
        tb_writer.add_scalar("test/he_rot_max", metrics["max_q_he"], iter_no)

        if eval_rel_pose:
            tb_writer.add_scalar("test/rel_trans_median", np.median(t_loss_R), iter_no)
            tb_writer.add_scalar("test/rel_trans_mean", np.mean(t_loss_R), iter_no)
            tb_writer.add_scalar("test/rel_trans_max", np.max(t_loss_R), iter_no)
            tb_writer.add_scalar("test/rel_rot_median", np.median(q_loss_R), iter_no)
            tb_writer.add_scalar("test/rel_rot_mean", np.mean(q_loss_R), iter_no)
            tb_writer.add_scalar("test/rel_rot_max", np.max(q_loss_R), iter_no)

    return metrics


if __name__ == "__main__":
    seed_everything(config.seed)

    model = torch.load(MODEL_PATH)

    test_dataset = MVSDataset(image_folder=DATA_PATH)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    _ = eval(
        model=model,
        dataloader=test_dataloader,
        device=config.device,
        num_samples=NUM_SAMPLES,
        opt_iterations=1,
    )
