import numpy as np
from skspatial.objects import Plane, Points


def estimate_flatplane(xyz):
    points = Points(xyz)
    plane = Plane.best_fit(points)
    normal_z = np.asarray(plane.normal)
    origin_x = np.asarray([1, 0, 0])
    normal_y = np.cross(normal_z, origin_x)
    normal_y = normal_y / np.linalg.norm(normal_y)
    normal_x = np.cross(normal_y, normal_z)
    normal_x = normal_x / np.linalg.norm(normal_x)
    rotation_normal2origin = np.asarray([normal_x, normal_y, normal_z]).T
    translation_normal2origin = np.asarray(plane.point)

    transform_normal2origin = np.eye(4)
    transform_normal2origin[:3, :3] = rotation_normal2origin
    transform_normal2origin[:3, 3] = translation_normal2origin
    transform_origin2normal = np.linalg.inv(transform_normal2origin)
    return transform_origin2normal


def get_points_with_wings(xyz, offset):
    """Add more balanced points to trajectory xyz for plane fitting.
        The main purpose is to prevent ambiguious fitting when trajectory is
        almost stright line.

    Args:
        xyz (ndarray): shape(N, 3)
        offset (float): wings length.

    Returns:
        ndarray: shape(5N, 3), points with balanced wings
    """
    x_left_offset = xyz - np.array([[offset, 0, 0]])
    x_right_offset = xyz - np.array([[-offset, 0, 0]])
    y_left_offset = xyz - np.array([[0, offset, 0]])
    y_right_offset = xyz - np.array([[0, -offset, 0]])
    xyz = np.concatenate([xyz, x_left_offset, x_right_offset, y_left_offset, y_right_offset], axis=0)
    return xyz


def robust_estimate_flatplane(xyz, offset=0.8):
    """estimate flat plane from points. Assuming the points are mostly in xy plane.

    Args:
        xyz (ndarray): shape (N, 3) xyz points.
        offset (float, optional): wings length. Defaults to 0.8.

    Returns:
        ndarray: transform_normal2origin
    """
    xyz = get_points_with_wings(xyz, offset)
    return estimate_flatplane(xyz)
