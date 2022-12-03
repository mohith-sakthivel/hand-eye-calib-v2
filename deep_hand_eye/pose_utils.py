import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as R

import torch


def quaternion_angular_error(
    q1: np.ndarray, q2: np.ndarray, in_degrees: bool = True
) -> np.ndarray:
    """
    Calculate angular error between two quaternions
    """
    d = abs(np.sum(q2 * q1, axis=-1))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    theta = 2 * np.arccos(d)
    if in_degrees:
        theta = np.rad2deg(theta)

    return theta


def qlog(q: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies logarithm map to a quaternion
    """

    if isinstance(q, np.ndarray):
        theta = np.arccos(q[..., 0:1])
        normal = q[..., 1:] / np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    elif isinstance(q, torch.Tensor):
        theta = torch.arccos(q[..., 0:1])
        normal = q[..., 1:] / torch.linalg.norm(q[..., 1:], dim=-1, keepdims=True)

    log_q = theta * normal

    return log_q


def qexp(log_q: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies the exponential map to a log_quaternion
    """
    if isinstance(log_q, np.ndarray):
        n = np.linalg.norm(log_q, axis=-1, keepdims=True)
        q = np.hstack((np.cos(n), np.sinc(n / np.pi) * log_q))
    elif isinstance(log_q, torch.Tensor):
        n = torch.linalg.norm(log_q, dim=-1, keepdims=True)
        q = torch.hstack((torch.cos(n), torch.sinc(n / torch.pi) * log_q))

    return q


def invert_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """
    Invert a homogeneous transformation matrix
    """
    T = np.array(T)
    R, t = T[0:3, 0:3], T[0:3, 3]
    return np.r_[np.c_[R.T, -np.dot(R.T, t)], [[0, 0, 0, 1]]]


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    q = R.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]
    q *= np.sign(q[0])  # constrain to hemisphere

    return q


def transformation_matrix_to_xyz_quaternion(T: np.ndarray) -> np.ndarray:
    quat_pose = np.empty(7)
    quat_pose[:3] = T[:3, 3]
    quat_pose[3:] = rotation_matrix_to_quaternion(T[:3, :3])

    return quat_pose


def transformation_matrix_to_xyz_log_quaternion(T: np.ndarray) -> np.ndarray:
    quat_pose = np.empty(6)
    quat_pose[:3] = T[:3, 3]
    quat_pose[3:] = qlog(rotation_matrix_to_quaternion(T[:3, :3]))

    return quat_pose


def xyz_quaternion_to_xyz_log_quaternion(xyz_quaternion: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    xyz = xyz_quaternion[..., :3]
    log_quaternion = qlog(xyz_quaternion[..., 3:])

    if isinstance(xyz, torch.Tensor):
        xyz_log_quaternion = torch.cat([xyz, log_quaternion], dim=-1)
    else:
        xyz_log_quaternion = np.concatenate([xyz, log_quaternion], dim=-1)

    return xyz_log_quaternion

def quaternion_to_axis_angle(q, in_degrees: bool = False):
    """
    Convert quaternion into axis angle representation
    """
    axis = q[..., 1:] / np.linalg.norm(q[..., 1:])
    angle = 2 * np.arccos(q[..., 0])

    if in_degrees:
        angle = np.rad2deg(angle)

    return axis, angle
