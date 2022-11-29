import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import deep_hand_eye.pose_utils as p_utils
from deep_hand_eye.utils import AttrDict
from deep_hand_eye.dataset import MVSDataset


MODEL_PATH = 'models/2022-10-17--19-11-31/model.pt'
DATA_PATH = "data/DTU_MVS_2014/Rectified/test/"
MAX_SAMPLES = 100
LOG_DIR = "eval"
LOG_IMAGES = True


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.batch_size = 1


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_data():
    seed_everything(config.seed)
    log_dir = Path(LOG_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = torch.load(MODEL_PATH)
    model.eval()

    test_dataset = MVSDataset(image_folder=DATA_PATH, get_raw_images=True)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    save_data = {
        "he_actual": [],
        "he_pred": [],
        "rel_ee": [],
        "rel_cam_actual": [],
        "rel_cam_pred": [],
        "edge_idx": []
    }

    # inference loop
    for i, data in enumerate(tqdm(test_dataloader, total=MAX_SAMPLES // config.batch_size)):
        if i == MAX_SAMPLES // config.batch_size:
            break

        data = data.to(config.device)
        output_he, output_R, _ = model(data)

        output_he = output_he.cpu().numpy()
        target_he = data.y.cpu().numpy()
        output_R = output_R.cpu().numpy()
        target_R = data.y_edge.cpu().numpy()

        rel_ee_motion = data.edge_attr.cpu().numpy()
        q = [p_utils.qexp(p[3:]) for p in rel_ee_motion]
        rel_ee_motion = np.hstack((rel_ee_motion[:, :3], np.asarray(q)))

        # Generate poses from predictions and label for hand-eye pose
        q = [p_utils.qexp(p[3:]) for p in output_he]
        output_he = np.hstack((output_he[:, :3], np.asarray(q)))
        q = [p_utils.qexp(p[3:]) for p in target_he]
        target_he = np.hstack((target_he[:, :3], np.asarray(q)))

        # Generate poses from predictions and label for relative poses
        q = [p_utils.qexp(p[3:]) for p in output_R]
        output_R = np.hstack((output_R[:, :3], np.asarray(q)))
        q = [p_utils.qexp(p[3:]) for p in target_R]
        target_R = np.hstack((target_R[:, :3], np.asarray(q)))
        
        save_data["he_actual"].append(target_he)
        save_data["he_pred"].append(output_he)
        save_data["rel_ee"].append(rel_ee_motion)
        save_data["rel_cam_actual"].append(target_R)
        save_data["rel_cam_pred"].append(output_R)
        save_data["edge_idx"].append(data.edge_index.cpu().numpy())

    for k, v in save_data.items():
        save_data[k] = np.array(v).squeeze()

    with open("temp/sample_data.pkl", "wb") as f:
        pickle.dump(save_data, f)


if __name__ == "__main__":
    generate_data()
