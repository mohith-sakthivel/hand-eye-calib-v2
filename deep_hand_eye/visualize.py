import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import open3d as o3d

import torch
from torch_geometric.loader import DataLoader

from deep_hand_eye.pose_utils import quaternion_angular_error, qexp, quaternion_to_axis_angle, transformation_matrix_to_xyz_quaternion
from deep_hand_eye.utils import AttrDict
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.viz_utils import vizualize_poses

config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.batch_size = 2
config.log_images = True
config.log_dir = Path("eval")
config.max_samples = 20
config.data_path = "data/DTU_MVS_2014/Rectified/test/"
config.model_dir = Path("models")
config.model_name = "2022-10-17--19-11-31"
config.num_hist_bins = 125
config.num_images_per_graph = 5
config,show_images = True
# Add config.model_checkpoint here after it's been implemented in train.py


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelEval(object):
    def __init__(self, config: AttrDict) -> None:
        self.config = config
        
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        
        self.model = torch.load(os.path.join(config.model_dir, config.model_name, "model.pt"))
        self.test_dataset = MVSDataset(image_folder=config.data_path, get_raw_images=True)
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

        self.trans_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt, axis=-1)
        self.quat_criterion = quaternion_angular_error

    @staticmethod
    def get_stats(qty):
        return np.median(qty, axis=0), np.mean(qty, axis=0), np.max(qty, axis=0)
    
    def plot_statistics(self, trans_error_he, quat_error_he, trans_error_rel, quat_error_rel):
        error_names = ["Hand-eye Translational Error", "Hand-eye Rotational Error", 
                "Relative Translational Error", "Relative Rotational Error"]
        errors = [trans_error_he, quat_error_he, trans_error_rel, quat_error_rel]

        fig, axs = plt.subplots(2, 2, tight_layout=True)
        for row in range(2):
            for col in range(2):
                idx = row * 2 + col
                unit = "deg" if col == 1 else "mm"
                axs[row, col].hist(1000 * errors[idx], bins=self.config.num_hist_bins)
                axs[row, col].set_title(error_names[idx])
                axs[row, col].set_xlabel(f'Error [{unit}]')
                axs[row, col].set_ylabel('Number of Instances')
                axs[row, col].minorticks_on()

                mean, median, max = ModelEval.get_stats(errors[idx])
                print(f'{error_names[idx]}: \n'
                f'\t median {median:3.2f} {unit} \n'
                f'\t mean {mean:3.2f} {unit} \n'
                f'\t max {max:3.2f} {unit} \n'
        
        plt.savefig(str(self.config.log_dir / f"{self.config.model_name}_histogram.jpg"))
        if self.config.show_images:
            plt.show()
    
    @staticmethod
    def maximum_axis_difference(axes):
        max_axis_diff = 0
        for i in range(axes.shape[0]):
            for j in np.arange(i+1, axes.shape[0]):
                axis_diff = np.abs(np.arccos(np.dot(axes[i], axes[j])))
                if axis_diff > max_axis_diff:
                    max_axis_diff = axis_diff
        return max_axis_diff

    @staticmethod
    def smallest_enclosing_sphere(points):
        # Surprisingly challenging lol
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        oriented_bb = pcd.get_oriented_bounding_box()
        return np.linalg.norm(oriented_bb.extent)

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        trans_error_he = []
        quat_error_he = []
        trans_error_rel = []
        quat_error_rel = []
        count = 0

        # inference loop
        for data in tqdm(self.test_dataloader, total=self.config.max_samples // self.config.batch_size):
            data = data.to(config.device)
            output_he, output_rel, _ = self.model(data)

            # Move to local CPU
            output_he = output_he.cpu().numpy()
            target_he = data.y.cpu().numpy()
            output_rel = output_rel.cpu().numpy()
            target_rel = data.y_edge.cpu().numpy()

            # Generate poses from predictions and label for hand-eye pose
            output_he = np.array([transformation_matrix_to_xyz_quaternion(he_pose) for he_pose in output_he])
            target_he = np.array([transformation_matrix_to_xyz_quaternion(he_pose) for he_pose in target_he])

            # Calculate hand-eye losses
            trans_error_he_batch = self.trans_criterion(output_he[:, :3], target_he[:, :3])
            quat_error_he_batch = self.quat_criterion(output_he[:, 3:], target_he[:, 3:])

            # Generate poses from predictions and label for relative poses
            output_rel_quat = [qexp(p[3:]) for p in output_rel]
            output_rel = np.hstack((output_rel[:, :3], np.asarray(output_rel_quat)))
            target_rel_quat = [qexp(p[3:]) for p in target_rel]
            target_rel = np.hstack((target_rel[:, :3], np.asarray(target_rel_quat)))

            # Compute translational and angular error
            trans_error_rel_batch = self.trans_criterion(output_rel[:, :3], target_rel[:, :3])
            quat_error_rel_batch = self.quat_criterion(output_rel[:, 3:], target_rel[:, 3:])

            # Aggregate errors
            trans_error_he.append(trans_error_he_batch)
            quat_error_he.append(quat_error_he_batch)
            trans_error_rel.append(trans_error_rel_batch)
            quat_error_rel.append(quat_error_rel_batch)

            # ?
            edge_batch_idx = data.batch[data.edge_index[0]].cpu().numpy()
            edge_first_nodes = data.edge_index[0].cpu().numpy()

            # Visualization of Results
            for graph_id in range(data.num_graphs):
                img_list = data.raw_images[graph_id]
                count += 1

                edge_mask = edge_batch_idx == graph_id
                rel_pos_error = trans_error_rel_batch[edge_mask].mean(axis=0) * 1000
                rel_rot_error = quat_error_rel_batch[edge_mask].mean(axis=0)

                # Compute maximum axis angle deviation
                input_rel_rotation_axes = [q[1:] / np.linalg.norm(q[1:]) for q in target_rel[edge_mask, 3:]]
                max_axis_diff = maximum_axis_difference(np.array(input_rel_rotation_axes))

                # Compute minimum sphere size
                min_sphere_radius = smallest_enclosing_sphere(target_rel[edge_mask, :3])

                num_rows = np.floor(np.sqrt(len(img_list)))
                num_cols = np.ceil(len(img_list)/num_rows)

                # Add logic to make custom # nodes per image, currently doesn't work
                rows = list()
                for row in range(num_rows):
                    row = list()
                    for col in range(num_cols):
                        idx = row * num_cols + num_cols
                        row.append(cv2.resize(img_list[idx], (400, 300)))
                    rows.append(cv2.hconcat(row))
                cv2.vconcat(rows)

                axis, angle = quaternion_to_axis_angle(target_he[graph_id, 3:], in_degrees=True)

                pos_str = "  ".join([f"{1000 * item: 0.2f}" for item in target_he[graph_id, :3]])
                he_pos_str = "HE Translation (mm):  " + pos_str + f"  Total: {1000 * np.linalg.norm(target_he[graph_id, :3]): 0.2f}"
                axis_str = "  ".join([f"{item: 0.4f}" for item in axis])
                he_rot_str = f"HE Angle {angle: 0.2f} degrees   Axis: " + axis_str

                cv2.putText(combined, he_pos_str, (400*num_cols, 550), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                cv2.putText(combined, he_rot_str, (400*num_cols, 600), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                cv2.putText(combined, f"HE Translation Error: {trans_error_he_batch[graph_id]*1000:3.2f} mm", (400*num_cols, 650), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                cv2.putText(combined, f"HE Rotation Error: {quat_error_he_batch[graph_id]:3.2f} degrees", (400*num_cols, 700), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                cv2.putText(combined, f"Relative Translation Error: {rel_pos_error:3.2f} mm", (400*num_cols, 750), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                cv2.putText(combined, f"Relative Rotation Error: {rel_rot_error:3.2f} degrees", (400*num_cols, 800), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
                
                if self.config.log_images:
                    plt.imsave(str(self.log_dir / f"data_{count}.jpg"), combined)
                if self.config.show_images:
                    plt.imshow(combined)
                    plt.show()

                graph_first_node = min(edge_first_nodes[edge_mask])
                reqd_edges = edge_first_nodes == graph_first_node

                rel_poses = target_rel[reqd_edges]
                rel_poses = np.vstack((np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=float), rel_poses))
                vizualize_poses(rel_poses)

            if count > self.max_samples:
                break

        trans_error_he = np.array(np.concatenate(trans_error_he, axis=0))
        quat_error_he = np.array(np.concatenate(quat_error_he, axis=0))
        trans_error_rel = np.array(np.concatenate(trans_error_rel, axis=0).reshape(-1))
        quat_error_rel = np.array(np.concatenate(quat_error_rel, axis=0).reshape(-1))

        ModelEval.plot_statistics(self, trans_error_he, quat_error_he, trans_error_rel, quat_error_rel)

def main():
    seed_everything(config.seed)
    model_eval = ModelEval(config=config)
    model_eval.eval()

if __name__ == "__main__":
    main()
