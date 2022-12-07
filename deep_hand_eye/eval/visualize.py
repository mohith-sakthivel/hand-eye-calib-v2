import os
import cv2
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from deep_hand_eye.pose_utils import (
    quaternion_angular_error,
    qexp,
    quaternion_to_axis_angle,
    transformation_matrix_to_xyz_quaternion,
)
from deep_hand_eye.utils import AttrDict
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.eval.utils import visualize_poses

config = AttrDict()
config.seed = 0
config.device = "cuda"
config.opt_iterations = 2
config.num_workers = 0
config.batch_size = 8
config.log_images = False
config.log_dir = Path("viz_data")
config.max_samples = 100
config.data_path = "data/DTU_MVS_2014/Rectified/test/"
config.model_dir = Path("models")
config.model_name = "11_anton_conv_self_attn_after_refactor"
config.num_hist_bins = 125
config.show_images = True
config.per_image_width = 480
config.per_image_height = 360


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EvaluateModel(object):
    def __init__(self, config: AttrDict) -> None:
        self.config = config

        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        self.model = torch.load(config.model_dir / config.model_name / "model.pt")
        self.test_dataset = MVSDataset(image_folder=config.data_path, get_viz_data=True)
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        self.trans_criterion = lambda t_pred, t_gt: np.linalg.norm(
            t_pred - t_gt, axis=-1
        )
        self.quat_criterion = quaternion_angular_error

    @staticmethod
    def get_stats(qty):
        return np.mean(qty, axis=0), np.median(qty, axis=0), np.max(qty, axis=0)

    def plot_statistics(
        self, trans_error_he, quat_error_he, trans_error_rel, quat_error_rel
    ):
        error_names = [
            "Hand-eye Translational Error",
            "Hand-eye Rotational Error",
            "Relative Translational Error",
            "Relative Rotational Error",
        ]
        errors = [trans_error_he, quat_error_he, trans_error_rel, quat_error_rel]

        fig, axs = plt.subplots(2, 2, tight_layout=True)
        for row in range(2):
            for col in range(2):
                idx = row * 2 + col
                unit = "deg" if col == 1 else "mm"
                axs[row, col].hist(1000 * errors[idx], bins=self.config.num_hist_bins)
                axs[row, col].set_title(error_names[idx])
                axs[row, col].set_xlabel(f"Error [{unit}]")
                axs[row, col].set_ylabel("Number of Instances")
                axs[row, col].minorticks_on()

                mean, median, max = EvaluateModel.get_stats(errors[idx])
                print(
                    f"{error_names[idx]}: \n"
                    f"\t median {median:3.2f} {unit} \n"
                    f"\t mean {mean:3.2f} {unit} \n"
                    f"\t max {max:3.2f} {unit} \n"
                )

        plt.savefig(
            str(self.config.log_dir / f"{self.config.model_name}_histogram.jpg")
        )
        if self.config.show_images:
            plt.show()

    @staticmethod
    def maximum_axis_difference(axes):
        max_axis_diff = 0
        for i in range(axes.shape[0]):
            for j in np.arange(i + 1, axes.shape[0]):
                axis_diff = np.arccos(np.abs(np.clip(np.dot(axes[i], axes[j]), -1, 1)))
                if axis_diff > max_axis_diff:
                    max_axis_diff = axis_diff
        return max_axis_diff

    @staticmethod
    def smallest_enclosing_sphere(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        oriented_bb = pcd.get_oriented_bounding_box()
        return np.linalg.norm(oriented_bb.extent) / 2

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        trans_error_he = []
        quat_error_he = []
        trans_error_rel = []
        quat_error_rel = []
        count = 0

        # inference loop
        for data in tqdm(
            self.test_dataloader,
            total=self.config.max_samples // self.config.batch_size,
        ):
            self.test_dataloader.dataset.reset_indices()
            data = data.to(config.device)
            output_he, output_rel, _ = self.model(data, opt_iterations=config.opt_iterations)

            # Move to local CPU
            output_he = output_he.cpu().numpy()
            target_he = data.y.cpu().numpy()
            output_rel = output_rel.cpu().numpy()
            target_rel = data.y_edge.cpu().numpy()
            camera_poses = np.array(data.camera_poses)

            # Generate poses from predictions and label for hand-eye pose
            output_he = np.array(
                [
                    transformation_matrix_to_xyz_quaternion(he_pose)
                    for he_pose in output_he
                ]
            )
            target_he = np.array(
                [
                    transformation_matrix_to_xyz_quaternion(he_pose)
                    for he_pose in target_he
                ]
            )

            # Calculate hand-eye losses
            trans_error_he_batch = self.trans_criterion(
                output_he[:, :3], target_he[:, :3]
            )
            quat_error_he_batch = self.quat_criterion(
                output_he[:, 3:], target_he[:, 3:]
            )

            # Turn output pose rotation into quaternion
            output_rel_quat = [qexp(p[3:]) for p in output_rel]
            output_rel = np.hstack((output_rel[:, :3], np.asarray(output_rel_quat)))

            # Compute translational and angular error
            trans_error_rel_batch = self.trans_criterion(
                output_rel[:, :3], target_rel[:, :3]
            )
            quat_error_rel_batch = self.quat_criterion(
                output_rel[:, 3:], target_rel[:, 3:]
            )

            print("Translation:\n", trans_error_he_batch)
            print("Rotation:\n", quat_error_he_batch)

            # Aggregate errors
            trans_error_he.append(trans_error_he_batch)
            quat_error_he.append(quat_error_he_batch)
            trans_error_rel.append(trans_error_rel_batch)
            quat_error_rel.append(quat_error_rel_batch)

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
                input_rel_rotation_axes = [
                    q[1:] / np.linalg.norm(q[1:]) for q in target_rel[edge_mask, 3:]
                ]
                max_axis_diff = EvaluateModel.maximum_axis_difference(
                    np.array(input_rel_rotation_axes)
                )

                # Compute minimum sphere size
                min_sphere_radius = EvaluateModel.smallest_enclosing_sphere(
                    camera_poses[graph_id, :, :3, 3]
                )

                num_rows = int(np.floor(np.sqrt(len(img_list))))
                num_cols = int(
                    (
                        np.ceil(len(img_list) / num_rows)
                        if len(img_list) % num_rows == 1
                        else np.ceil(len(img_list) / num_rows) + 1
                    )
                )

                # Add logic to make custom # nodes per image, currently doesn't work
                rows_images = list()
                for row in range(num_rows):
                    row_images = list()
                    for col in range(num_cols):
                        idx = row * num_cols + col
                        if idx < len(img_list):
                            row_images.append(
                                cv2.resize(
                                    img_list[idx],
                                    (
                                        self.config.per_image_width,
                                        self.config.per_image_height,
                                    ),
                                )
                            )
                        else:
                            row_images.append(
                                cv2.resize(
                                    np.zeros_like(img_list[0]),
                                    (
                                        self.config.per_image_width,
                                        self.config.per_image_height,
                                    ),
                                )
                            )
                    rows_images.append(cv2.hconcat(row_images))
                viz_image = cv2.vconcat(rows_images)

                axis, angle = quaternion_to_axis_angle(
                    target_he[graph_id, 3:], in_degrees=True
                )

                pos_str = "  ".join(
                    [f"{1000 * item: 0.2f}" for item in target_he[graph_id, :3]]
                )
                he_pos_str = (
                    "HE Translation (mm):  "
                    + pos_str
                    + f"  Total: {1000 * np.linalg.norm(target_he[graph_id, :3]): 0.2f}"
                )
                axis_str = "  ".join([f"{item: 0.4f}" for item in axis])
                he_rot_str = f"HE Angle {angle: 0.2f} degrees   Axis: " + axis_str

                cv2.putText(
                    viz_image,
                    he_pos_str,
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    he_rot_str,
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 60,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    f"HE Translation Error: {trans_error_he_batch[graph_id]*1000:3.2f} mm",
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 90,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    f"HE Rotation Error: {quat_error_he_batch[graph_id]:3.2f} degrees",
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 120,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    f"Relative Translation Error: {rel_pos_error:3.2f} mm",
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 150,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    f"Relative Rotation Error: {rel_rot_error:3.2f} degrees",
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 180,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    f"Maximum Rotation Axis Difference {max_axis_diff:3.2f} rad",
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 210,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    viz_image,
                    f"Smallest Enclosing Sphere Radius: {min_sphere_radius:3.2f} m",
                    (
                        self.config.per_image_width * (num_cols - 1) + 10,
                        self.config.per_image_height * (num_rows - 1) + 240,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )

                if self.config.log_images:
                    plt.imsave(
                        str(self.config.log_dir / f"data_{count}.jpg"), viz_image
                    )
                if self.config.show_images:
                    viz_image = cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("data", viz_image)
                    cv2.waitKey(0)
                    cv2.destroyWindow("data")

                graph_first_node = min(edge_first_nodes[edge_mask])
                reqd_edges = edge_first_nodes == graph_first_node

                rel_poses = target_rel[reqd_edges]
                rel_poses = np.vstack(
                    (np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=float), rel_poses)
                )
                if config.show_images:
                    visualize_poses(rel_poses)

            if count > self.config.max_samples:
                break

        trans_error_he = np.array(np.concatenate(trans_error_he, axis=0))
        quat_error_he = np.array(np.concatenate(quat_error_he, axis=0))
        trans_error_rel = np.array(np.concatenate(trans_error_rel, axis=0).reshape(-1))
        quat_error_rel = np.array(np.concatenate(quat_error_rel, axis=0).reshape(-1))

        EvaluateModel.plot_statistics(
            self, 1000 * trans_error_he, quat_error_he, 1000 * trans_error_rel, quat_error_rel
        )


def main():
    seed_everything(config.seed)
    evaluation_object = EvaluateModel(config=config)
    evaluation_object.evaluate()


if __name__ == "__main__":
    main()
