import cv2
import torch
import numpy as np


def createFlatMesh(x_length, y_length, resolution=0.1):
    """
    Create a flat mesh for testing.

    Args:
        x_length (float): Length along x of the mesh.
        y_length (float): Length along y of the mesh.
        resolution (float): Resolution of the mesh.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing the vertices of the mesh.
        torch.Tensor: A tensor of shape (N, 3) containing the faces of the mesh.
    """
    num_vertices_x = int(x_length / resolution) + 1
    num_vertices_y = int(y_length / resolution) + 1
    assert num_vertices_x > 0 and num_vertices_y > 0, "Mesh resolution too high."

    vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
    vertices[:, :, 0] = torch.unsqueeze(torch.linspace(0, x_length, num_vertices_x), dim=0).T
    vertices[:, :, 1] = torch.unsqueeze(torch.linspace(0, y_length, num_vertices_y), dim=0)
    vertices = vertices.reshape(-1, 3)

    # 2 means top-right and bottom-left triangles
    # 3 means 3 vertices of each trianle
    faces = torch.zeros((num_vertices_x - 1, num_vertices_y - 1, 2, 3), dtype=torch.int64)
    all_indices = torch.arange(0, num_vertices_x * num_vertices_y, 1, dtype=torch.int64).reshape((num_vertices_x, num_vertices_y))
    faces[:, :, 0, 0] = all_indices[:-1, :-1]
    faces[:, :, 0, 1] = all_indices[:-1, 1:]
    faces[:, :, 0, 2] = all_indices[1:, 1:]
    faces[:, :, 1, 0] = all_indices[:-1, :-1]
    faces[:, :, 1, 1] = all_indices[1:, 1:]
    faces[:, :, 1, 2] = all_indices[1:, :-1]
    faces = faces.reshape(-1, 3)
    return vertices, faces, (num_vertices_x, num_vertices_y)


def createHiveFlatMesh(x_length, y_length, resolution=0.1):
    """
    Create a flat hive mesh.

    Args:
        x_length (float): Length along x of the mesh.
        y_length (float): Length along y of the mesh.
        poses (torch.Tensor): A tensor of shape (N, 3) containing the poses of the mesh.
        resolution (float): Resolution of the mesh. default: 1
    Returns:
        torch.Tensor: A tensor of shape ((num_vertices_x * num_vertices_y), 3) containing the vertices of the mesh.
        torch.Tensor: A tensor of shape ((num_vertices_x-1) * (num_vertices_y-1), 3) containing the faces of the mesh.
    """
    x_resolution = resolution
    y_resolution = x_resolution * 2 / 1.7320508075688772
    num_vertices_x = int(x_length / x_resolution) + 1
    num_vertices_y = int(y_length / y_resolution) + 1
    assert num_vertices_x > 0 and num_vertices_y > 0, "Mesh resolution too high."
    vertices = torch.zeros((num_vertices_x, num_vertices_y, 3), dtype=torch.float32)
    vertices[:, :, 0] = torch.unsqueeze(torch.linspace(0, x_length, num_vertices_x), dim=0).T
    for i in range(num_vertices_x):
        if i % 2 == 0:
            vertices[i, :, 1] = torch.linspace(0, y_length + y_resolution / 2, num_vertices_y)
        else:
            vertices[i, :, 1] = torch.linspace(-y_resolution / 2, y_length, num_vertices_y)
    vertices = vertices.reshape(-1, 3)

    # 2 means top-right and bottom-left triangles
    # 3 means 3 vertices of each trianle
    faces = torch.zeros((num_vertices_x - 1, num_vertices_y - 1, 2, 3), dtype=torch.int64)
    all_indices = torch.arange(0, num_vertices_x * num_vertices_y, 1, dtype=torch.int64).reshape((num_vertices_x, num_vertices_y))
    faces[:, :, 0, 0] = all_indices[:-1, :-1]
    faces[:, :, 0, 1] = all_indices[:-1, 1:]
    faces[:, :, 0, 2] = all_indices[1:, 1:]
    faces[:, :, 1, 0] = all_indices[:-1, :-1]
    faces[:, :, 1, 1] = all_indices[1:, 1:]
    faces[:, :, 1, 2] = all_indices[1:, :-1]

    # face 0 vert 0 down, face 1 vert 1 up in odd row
    faces[1::2, :, 0, 0] = faces[1::2, :, 1, 2]
    faces[1::2, :, 1, 1] = faces[1::2, :, 0, 1]
    faces = faces.reshape(-1, 3)
    return vertices, faces, (num_vertices_x, num_vertices_y)


def cutHiveMeshWithPoses(vertices, faces, bev_size_pixel, x_length, y_length, poses_xy, resolution=0.1, cut_range=30):
    """
    Cut mesh using poses

    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) containing the vertices of the mesh.
        faces (torch.Tensor): A tensor of shape (N, 3) containing the faces of the mesh.
        bev_size_pixel (tuple): The size of the bev in pixel.
        x_length (float): Length along x of the mesh.
        y_length (float): Length along y of the mesh.
        poses_xy (torch.Tensor): A tensor of shape (N, 2) containing the poses in camera2world transform.
    """
    import pymeshlab
    x_resolution = resolution
    y_resolution = x_resolution * 2 / 1.7320508075688772
    (num_vertices_x, num_vertices_y) = bev_size_pixel
    # pose_xy to pixel_xy
    pixel_xy = np.zeros_like(poses_xy)
    pixel_xy[:, 0] = (x_length / 2 - poses_xy[:, 0]) / x_resolution
    pixel_xy[:, 1] = (y_length / 2 - poses_xy[:, 1]) / y_resolution
    pixel_xy = np.unique(pixel_xy.round(), axis=0)

    # construct the mask
    mask = np.zeros((num_vertices_x - 1, num_vertices_y - 1), dtype=np.uint8)
    pixel_xy[:, 0] = np.clip(pixel_xy[:, 0], 0, num_vertices_x - 2)
    pixel_xy[:, 1] = np.clip(pixel_xy[:, 1], 0, num_vertices_y - 2)
    pixel_xy = pixel_xy.astype(np.long)
    mask[pixel_xy[:, 0], pixel_xy[:, 1]] = 1
    mask = mask[::-1, ::-1]  # rotate the mask 180 degrees
    # cv2.imwrite('mask.png', mask.astype(np.uint8) * 255)

    # dilate the mask
    kernel_size = int(cut_range / resolution)  # around cut_range meters
    kernel = np.ones((kernel_size, kernel_size), dtype=np.long)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    # cv2.imwrite('mask_dilate.png', mask.astype(np.uint8) * 255)

    # give faces colors
    face_quality = np.ones((num_vertices_x - 1, num_vertices_y - 1, 2, 1), dtype=np.float64)
    face_quality[mask == 0, :, 0] = 0.0
    face_quality = face_quality.reshape(-1, 1)
    source_mesh = pymeshlab.Mesh(vertex_matrix=vertices.numpy(), face_matrix=faces.numpy(), f_quality_array=face_quality)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(source_mesh, "source_mesh")
    m = ms.current_mesh()
    # face_color_matrix = m.face_color_matrix()
    ms.conditional_face_selection(condselect="fq < 1")  # equals to fr == 0
    # print(ms.current_mesh().selected_face_number())
    ms.delete_selected_faces()
    ms.remove_unreferenced_vertices()
    m = ms.current_mesh()

    # get numpy arrays of vertices and faces of the current mesh
    v_matrix = torch.from_numpy(m.vertex_matrix().astype(np.float32))
    f_matrix = torch.from_numpy(m.face_matrix().astype(np.int64))
    # ms.save_current_mesh("filted.ply")

    return v_matrix, f_matrix, (num_vertices_x, num_vertices_y)


def fps_by_distance(pointcloud, min_distance, return_idx=True, allow_same_gps=False):
    """subsample pointcloud by furthest point sampling algorithm

    Args:
        pointcloud (ndarray): shape=[N, 3] or [N, 2]
        min_distance (float): meters, minimum distance allowed in subsampled pointcloud
        return_idx (bool, optional): If set to true, return sampling index of original pointclouds.
        Defaults to True. Otherwise, return subsampled pointcloud
    """
    assert 2 <= pointcloud.shape[1] <= 3
    num_points = pointcloud.shape[0]
    sample_idx = np.zeros(num_points, dtype=bool)
    start_idx = np.random.randint(0, num_points)
    sample_idx[start_idx] = True
    sampled_min_distance = 1e9

    while np.any(~sample_idx) and sampled_min_distance > min_distance:
        sampled_points = pointcloud[sample_idx]
        local_min_list = []
        for point in pointcloud:
            distance = np.linalg.norm(point - sampled_points, ord=np.inf, axis=1)
            local_min = np.min(distance)
            if allow_same_gps and local_min == 0:
                local_min = min_distance + 1
            local_min_list.append(local_min)
        local_min_array = np.array(local_min_list)
        local_min_array[sample_idx] = 0
        furthest_point_idx = np.argmax(local_min_array)
        sampled_min_distance = local_min_array[furthest_point_idx]
        sample_idx[furthest_point_idx] = sampled_min_distance > min_distance
    if return_idx:
        return sample_idx
    else:
        return pointcloud[sample_idx]


if __name__ == '__main__':
    # Create a mesh.
    vertices, faces, bev_size_pixel = createHiveFlatMesh(1.0, 1.0)
    pass
