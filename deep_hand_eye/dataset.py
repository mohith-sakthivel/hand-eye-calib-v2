import json
import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch_geometric.data import Data
from PIL import Image
from glob import glob
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm

class MVSDataset(Dataset):
    def __init__(self, image_folder="Data/DTU_MVS_2014/Rectified/",
                 json_file="Data/DTU_MVS_2014/camera_pose.json", num_nodes=5,
                 max_trans_offset=0.1, max_rot_offset=0, transform=None, image_size=(256, 256)):
        # Storing the image folder
        self.image_folder = image_folder

        # Initialize idx used to loop through the folders
        self.folder_idx = 0

        # Get all scene names
        self.scans = os.listdir(image_folder)

        # Number of nodes in the graph
        self.num_nodes = int(num_nodes)

        # Camera positions dictionary
        self.camera_positions = json.load(open(json_file, 'r'))

        # Translational and rotational magnitude
        self.max_trans_offset = max_trans_offset
        self.max_rot_offset = max_rot_offset

        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # Image formats for the different brightnesses
        self.image_format = ['rect_{}_0_r5000.png', 'rect_{}_1_r5000.png', 'rect_{}_2_r5000.png',
                            'rect_{}_3_r5000.png', 'rect_{}_4_r5000.png', 'rect_{}_5_r5000.png',
                            'rect_{}_6_r5000.png', 'rect_{}_max.png']

        # Create edge index tensor
        self.edge_index = edge_idx_gen(self.num_nodes)

        # Mapping between idx and unique images
        self.idx_to_img_mapping = idx_to_img_map(image_folder)
        
    def __len__(self):
        return len(self.idx_to_img_mapping)

    def __getitem__(self, idx):
        """
        returns:
            image_list : list of images corresponding to those sampled from the given scene
            transforms : nxnx7 tensor capturing relative transforms between n sampled images
                         each entry contains 7 dim vector: [translation (1x3) | quaternion (1x4)]
                         [x y z | w x y z]
            hand_eye : hand eye translation vector used to generate data
        """
        ### Hand-eye transform matrix
        # Random translational offset
        hand_trans = np.random.uniform(-self.max_trans_offset, self.max_trans_offset, 3)
        # Random rotational angle
        hand_rpy = np.random.uniform(-self.max_rot_offset, self.max_rot_offset, 3)
        # Roll matrix
        hand_roll = np.array([[1, 0, 0],
                            [0, np.cos(hand_rpy[0]), -np.sin(hand_rpy[0])],
                            [0, np.sin(hand_rpy[0]), np.cos(hand_rpy[0])]])
        # Pitch matrix
        hand_pitch = np.array([[np.cos(hand_rpy[1]), 0, np.sin(hand_rpy[1])],
                            [0, 1, 0],
                            [-np.sin(hand_rpy[1]), 0, np.cos(hand_rpy[1])]])
        # Yaw matrix
        hand_yaw = np.array([[np.cos(hand_rpy[2]), -np.sin(hand_rpy[2]), 0],
                            [np.sin(hand_rpy[2]), np.cos(hand_rpy[2]), 0],
                            [0, 0, 1]])
        # Combined rotational matrix
        hand_rotation = hand_yaw @ hand_pitch @ hand_roll

        # Hand-eye transformation matrix
        hand_eye_matrix = np.zeros((4,4))
        hand_eye_matrix[:3,3] = hand_trans
        hand_eye_matrix[:3,:3] = hand_rotation
        hand_eye_matrix[3,3] = 1
        # Get inverse to multiply onto absolute positions
        hand_eye_inv = hand_eye_matrix
        hand_eye_inv[:3,3] = -hand_eye_inv[:3,3]
        hand_eye_inv[:3,:3] = hand_rotation.T
        # Create ground truth 7-sized vector of the hand-eye calibration
        hand_eye = np.empty(7)
        hand_eye[:3] = hand_trans
        hand_eye[3] = R.from_matrix(hand_rotation).as_quat()[3]
        hand_eye[4:] = R.from_matrix(hand_rotation).as_quat()[:3]
        
        ### Get 5 images of a randomized brightness
        image_list = []
        # Tuple of (folder idx, image idx)
        folder_img = self.idx_to_img_mapping[idx]
        # Set folder directory - derived from idx
        folder = self.image_folder + self.scans[folder_img[0]] + '/'
        # Get initial image based on image number derived from idx
        img_name = os.listdir(folder)[folder_img[1]]
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
        image_names = glob(folder + img_format.format('*'))

        # Create index list
        index_list = []
        # Sample N images = num_nodes
        loop_var = 0
        while loop_var < self.num_nodes:
            index = random.choice(range(len(image_names)))
            if index not in index_list and index != chosen_idx:
                index_list.append(index)
                loop_var += 1

        # For each sampled index, get the image and append as a tensor
        for idx in index_list:
            # +1 because the dataset is 1-base rather than 0-base like Python indexing
            img_id = f'{idx+1:0>3}'
            file_name = folder + img_format.format(img_id)
            image = Image.open(file_name).convert('RGB')
            image = self.transform(image)
            image_list.append(image)
        image_list = torch.stack(image_list)

        # Initialize table for all relative transforms
        relative_ee_transforms = np.zeros((self.num_nodes, self.num_nodes, 7))
        relative_cam_transforms = np.zeros_like(relative_ee_transforms)
        # List of end effector poses
        ee_poses = []
        cam_poses = []
        # Obtain absolute end effector pose from the JSON file
        for index_idx in index_list:
            # Camera pose components from JSON
            abs_translation = np.array(self.camera_positions['pose']['trans'][index_idx])
            abs_rotation = np.array(self.camera_positions['pose']['rot'][index_idx])
            # Camera pose matrix
            abs_pose = np.zeros((4,4))
            abs_pose[:3,:3] = abs_rotation
            abs_pose[:3,3] = abs_translation.T
            abs_pose[3,3] = 1

            # Append camera pose
            cam_poses.append(abs_pose)

            # Use inverse of hand-eye matrix to get end effector pose
            ee_pose = hand_eye_inv @ abs_pose
            ee_poses.append(ee_pose)

        # Loop through randomly sampled positions
        # Format is [x y z | w x y z] for [translation | rotation]
        for from_idx in range(self.num_nodes):
            for to_idx in range(self.num_nodes):
                # If relative to itself
                if from_idx == to_idx:
                    # Set to zero transform quaternion
                    relative_ee_transforms[from_idx, to_idx, :3] = 0
                    relative_ee_transforms[from_idx, to_idx, 3] = 1
                    relative_ee_transforms[from_idx, to_idx, 4:] = 0

                    # Set to zero transform quaternion
                    relative_cam_transforms[from_idx, to_idx, :3] = 0
                    relative_cam_transforms[from_idx, to_idx, 3] = 1
                    relative_cam_transforms[from_idx, to_idx, 4:] = 0

                # If relative to another position
                else:
                    # For the end effector
                    rel_rotation = ee_poses[to_idx][:3, :3] @ ee_poses[from_idx][:3, :3].T 
                    rel_translation = ee_poses[to_idx][:3, 3] - ee_poses[from_idx][:3, 3]
                    # Obtain quaternion from relative rotation
                    rotation = R.from_matrix(rel_rotation)
                    rel_rotation_quaternion = rotation.as_quat()
                    # Populate relative transforms matrix
                    relative_ee_transforms[from_idx, to_idx, :3] = rel_translation
                    relative_ee_transforms[from_idx, to_idx, 3] = rel_rotation_quaternion[3]
                    relative_ee_transforms[from_idx, to_idx, 4:] = rel_rotation_quaternion[:3]

                    # For the camera pose
                    rel_rotation = cam_poses[to_idx][:3, :3] @ cam_poses[from_idx][:3, :3].T 
                    rel_translation = cam_poses[to_idx][:3, 3] - cam_poses[from_idx][:3, 3]
                    # Obtain quaternion from relative rotation
                    rotation = R.from_matrix(rel_rotation)
                    rel_rotation_quaternion = rotation.as_quat()
                    # Populate relative transforms matrix
                    relative_cam_transforms[from_idx, to_idx, :3] = rel_translation
                    relative_cam_transforms[from_idx, to_idx, 3] = rel_rotation_quaternion[3]
                    relative_cam_transforms[from_idx, to_idx, 4:] = rel_rotation_quaternion[:3]


        # Turn into a tensor
        relative_ee_transforms = torch.from_numpy(relative_ee_transforms)
        hand_eye = torch.from_numpy(hand_eye)

        print(self.edge_index.shape)

        # Create graph to return
        graph = Data(x=image_list, edge_index=self.edge_index, edge_attr=relative_ee_transforms,
                    y=hand_eye, y_edge=relative_cam_transforms)

        # Return as a dictionary
        return graph


def edge_idx_gen(num_nodes):
    edge_index_top = np.zeros(num_nodes*(num_nodes-1))
    """
    Generates the top and bottom array of edge indices separately and stacks them
    """
    for i in range(num_nodes):
        edge_index_top[i*(num_nodes-1):(i+1)*(num_nodes-1)] = i
    edge_index_low = np.zeros_like(edge_index_top)
    for i in range(num_nodes):
        index_idx = 0
        for j in range(num_nodes):
            if i != j:
                edge_index_low[index_idx+i*(num_nodes-1)] = j
                index_idx += 1
    edge_index = torch.tensor(np.stack([edge_index_top, edge_index_low]))
    return edge_index


def idx_to_img_map(image_folder):
    """
    Generates the unique mapping between every idx and the folder-image name index
    """
    # Find all scene folders
    scans = os.listdir(image_folder)
    img_list = []
    # Append a tuple of all scene and image positions
    print("Initializing dataset indices")
    for scan_idx in tqdm(range(len(scans))):
        scan_dir = image_folder + '/' + scans[scan_idx] + '/'
        img_names = os.listdir(scan_dir)
        for img_idx in range(len(img_names)):
            img_list.append((scan_idx, img_idx))

    # Return randomized list
    return img_list