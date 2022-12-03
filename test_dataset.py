import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from torch_geometric.loader import DataLoader

from deep_hand_eye.pose_utils import invert_transformation_matrix
from deep_hand_eye.dataset import MVSDataset


def test_dataset(test_size=64):

    dataset = MVSDataset(image_folder="data/DTU_MVS_2014/Rectified/train/")
    train_loader = DataLoader(dataset, batch_size=test_size, shuffle=True)
    data = next(iter(train_loader))
    print("Starting verificaiton...")

    for n in tqdm(range(data.num_graphs)):

        hand_eye = data[n].y.squeeze()
        E_T_C = np.vstack([hand_eye, [0, 0, 0, 1]])

        for i, (from_idx, to_idx) in enumerate(data[n].edge_index.T):

            E_T_E_transform = data[n].edge_attr[i]
            C_T_C_transform = data[n].y_edge[i]

            E_T_E_quat = E_T_E_transform[3:][[1, 2, 3, 0]]
            E_T_E_rotation_matrix = R.from_quat(E_T_E_quat).as_matrix()
            E_T_E_translate = np.array(E_T_E_transform[:3]).reshape(-1, 1)
            E_T_E = np.hstack([E_T_E_rotation_matrix, E_T_E_translate])
            E_T_E = np.vstack([E_T_E, [0, 0, 0, 1]])

            C_T_C_quat = C_T_C_transform[3:][[1, 2, 3, 0]]
            C_T_C_rotation_matrix = R.from_quat(C_T_C_quat).as_matrix()
            C_T_C_translate = np.array(C_T_C_transform[:3]).reshape(-1, 1)
            C_T_C = np.hstack([C_T_C_rotation_matrix, C_T_C_translate])
            C_T_C = np.vstack([C_T_C, [0, 0, 0, 1]])

            err_residuals = C_T_C - invert_transformation_matrix(E_T_C) @ E_T_E @ E_T_C
            assert np.all(np.abs(err_residuals) < 1e-5)


if __name__ == "__main__":
    test_dataset()
