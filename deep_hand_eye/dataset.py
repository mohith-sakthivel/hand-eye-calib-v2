import os
import json
import torch
import random
import numpy as np
from PIL import Image
from glob import glob
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset
from torch_geometric.data import Data
import torchvision.transforms as transforms

from deep_hand_eye.pose_utils import (
    invert_transformation_matrix,
    transformation_matrix_to_xyz_quaternion,
)


class MVSDataset(Dataset):
    def __init__(
        self,
        image_folder: str = "data/DTU_MVS_2014/Rectified/",
        camera_pose_json_file: str = "data/DTU_MVS_2014/camera_pose.json",
        num_nodes: List[int] = [5,],
        max_trans_offset: float = 0.3,
        max_rot_offset: float = 90.0,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple = (224, 224),
        get_eval_data: bool = False,
    ):

        self.image_folder = image_folder
        self.get_eval_data = get_eval_data
        self.num_nodes = num_nodes
        self.max_trans_offset = max_trans_offset
        self.max_rot_offset = np.deg2rad(max_rot_offset)
        self.camera_positions = json.load(open(camera_pose_json_file, "r"))

        # Setup Folders
        self.folder_idx = 0
        self.scans = os.listdir(image_folder)  # Get all scene names

        if transform == None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Image name formats for the different brightnesses
        self.image_format = [
            "rect_{}_0_r5000.png",
            "rect_{}_1_r5000.png",
            "rect_{}_2_r5000.png",
            "rect_{}_3_r5000.png",
            "rect_{}_4_r5000.png",
            "rect_{}_5_r5000.png",
            "rect_{}_6_r5000.png",
            "rect_{}_max.png",
        ]

        # Mapping between idx and unique images
        self.idx_to_img_mapping = self.idx_to_img_map()
        print("Dataset initialized...")

    def __len__(self) -> int:
        return len(self.idx_to_img_mapping)

    def __getitem__(self, idx) -> Data:
        """
        returns:
            image_list : list of images corresponding to those sampled from the given scene
            transforms : nxnx7 tensor capturing relative transforms between n sampled images
                         each entry contains 7 dim vector: [translation (1x3) | quaternion (1x4)]
                         [x y z | w x y z]
            hand_eye : hand eye translation vector used to generate data
        """

        def sample_unit_vector():
            vector = np.random.uniform(-1, 1, 3)
            return vector / np.linalg.norm(vector)

        # Random hand-eye transform matrix
        E_T_C_trans = np.random.uniform(0, self.max_trans_offset) * sample_unit_vector()
        E_T_C_angle = np.random.uniform(-self.max_rot_offset, self.max_rot_offset)
        E_T_C_axis = sample_unit_vector()
        E_T_C_rotation = R.from_rotvec(E_T_C_angle * E_T_C_axis).as_matrix()

        # Hand-eye transformation matrix
        E_T_C = np.zeros((4, 4))
        E_T_C[:3, 3] = E_T_C_trans
        E_T_C[:3, :3] = E_T_C_rotation
        E_T_C[3, 3] = 1
        C_T_E = invert_transformation_matrix(E_T_C)

        # Sample images of a particular random brightness
        image_list = []
        folder_idx, img_idx = self.idx_to_img_mapping[idx]
        # Set folder directory - derived from idx
        folder = self.image_folder + self.scans[folder_idx] + "/"
        # Get initial image based on image number derived from idx
        img_name = os.listdir(folder)[img_idx]
        # If brightness is within the indices of 0 to 7
        if len(img_name) == 20:
            brightness = int(img_name[9])
        else:
            brightness = 7
        # Get chosen index
        chosen_idx = int(img_name[5:8])
        # Choose the brightness - derived from initial image string
        img_format = self.image_format[brightness]
        # Find all images in the current scene with the selected brightness
        image_names = glob(folder + img_format.format("*"))

        # Create index list
        index_list = []
        # Sample N images
        curr_num_nodes = np.random.choice(self.num_nodes, 1)[0]
        # Create edge index tensor
        edge_index = edge_idx_gen(curr_num_nodes)

        loop_var = 0
        while loop_var < curr_num_nodes:
            index = random.choice(range(len(image_names)))
            if index not in index_list and index != chosen_idx:
                index_list.append(index)
                loop_var += 1

        extras = {}
        if self.get_eval_data:
            extras["raw_images"] = []

        # For each sampled index, get the image and append as a tensor
        for idx in index_list:
            # +1 because the dataset is 1-base rather than 0-base like Python indexing
            img_id = f"{idx+1:0>3}"
            file_name = folder + img_format.format(img_id)
            image = Image.open(file_name).convert("RGB")
            if self.get_eval_data:
                extras["raw_images"].append(np.asarray(image))
            transformed_image = self.transform(image)
            image_list.append(transformed_image)

        image_list = torch.stack(image_list)
        if self.get_eval_data:
            extras["raw_images"] = np.stack(extras["raw_images"], axis=0)

        # Initialize table for all relative transforms
        EA_T_EB_transforms = np.zeros((edge_index.shape[1], 7), dtype=np.float32)
        CA_T_CB_transforms = np.zeros_like(EA_T_EB_transforms)
        # List of end effector poses
        W_T_E_poses = []
        W_T_C_poses = []
        # Obtain absolute end effector pose from the JSON file
        for index_idx in index_list:
            # Camera pose matrix
            W_T_C = np.zeros((4, 4))
            W_T_C[:3, :3] = np.array(self.camera_positions["pose"]["rot"][index_idx])
            W_T_C[:3, 3] = np.array(self.camera_positions["pose"]["trans"][index_idx])
            W_T_C[3, 3] = 1
            W_T_C_poses.append(W_T_C)

            # Use inverse of hand-eye matrix to get end effector pose
            W_T_E = W_T_C @ C_T_E
            W_T_E_poses.append(W_T_E)

        if self.get_eval_data:
            extras["camera_poses"] = W_T_C_poses

        # Loop through randomly sampled positions
        # Format is [x y z | w x y z] for [translation | rotation]
        for from_idx in range(curr_num_nodes):
            index_idx = 0
            for to_idx in range(curr_num_nodes):
                # If relative to itself
                if from_idx == to_idx:
                    continue
                # Get rotation to put transform based on to/from indices
                edge_idx = index_idx + from_idx * (curr_num_nodes - 1)
                # For the end effector
                EA_T_EB = (
                    invert_transformation_matrix(W_T_E_poses[from_idx])
                    @ W_T_E_poses[to_idx]
                )
                EA_T_EB_transforms[edge_idx] = transformation_matrix_to_xyz_quaternion(
                    EA_T_EB
                )

                # For the camera pose
                CA_T_CB = (
                    invert_transformation_matrix(W_T_C_poses[from_idx])
                    @ W_T_C_poses[to_idx]
                )
                CA_T_CB_transforms[edge_idx] = transformation_matrix_to_xyz_quaternion(
                    CA_T_CB
                )

                index_idx += 1

        # Process required data to be returned
        E_T_C_tensor = E_T_C[..., :3, :].astype(np.float32)
        E_T_C_tensor = torch.from_numpy(E_T_C_tensor).unsqueeze(dim=0)
        EA_T_EB_tensors = torch.from_numpy(EA_T_EB_transforms)
        CA_T_CB_tensors = torch.from_numpy(CA_T_CB_transforms)

        # Create graph to return
        graph = Data(
            x=image_list,
            edge_index=edge_index,
            edge_attr=EA_T_EB_tensors,
            y=E_T_C_tensor,
            y_edge=CA_T_CB_tensors,
            **extras,
        )

        return graph

    def idx_to_img_map(
        self,
    ) -> List[Tuple[int, int]]:
        """
        Generates a list of all scan/image pairs. Used to assign a unique ID for each image
        in the dataset
        """
        # Find all scene folders
        scans = os.listdir(self.image_folder)
        img_list = []

        # Append a tuple of all scene and image positions
        for scan_idx in range(len(scans)):
            scan_dir = self.image_folder + "/" + scans[scan_idx] + "/"
            img_names = os.listdir(scan_dir)
            for img_idx in range(len(img_names)):
                img_list.append((scan_idx, img_idx))

        return img_list


def edge_idx_gen(num_nodes: int) -> torch.Tensor:
    """
    Generates the top and bottom array of edge indices separately and stacks them
    """
    edge_index_top = np.zeros(num_nodes * (num_nodes - 1), dtype=np.int64)
    for i in range(num_nodes):
        edge_index_top[i * (num_nodes - 1) : (i + 1) * (num_nodes - 1)] = i
    edge_index_low = np.zeros_like(edge_index_top)
    for i in range(num_nodes):
        index_idx = 0
        for j in range(num_nodes):
            if i != j:
                edge_index_low[index_idx + i * (num_nodes - 1)] = j
                index_idx += 1
    edge_index = torch.tensor(np.stack([edge_index_top, edge_index_low]))

    return edge_index
