import numpy as np
import cv2
import torch
from torch import nn
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.io import save_obj
from utils.renderer import SimpleShader
import pymeshlab


def mesh2height(mesh, bev_size_pixel):
    z_tensor = mesh._verts_list[0][:, 2]
    z_tensor = z_tensor.reshape(bev_size_pixel)
    return z_tensor


def loss2color(loss):
    min, max = loss.min(), loss.max()
    if (max-min) < 1e-7:
        loss = np.zeros_like(loss)
    else:
        # normalize depth by min max
        loss = (loss - min) / (max - min)
        loss.clip(0, 1)
    # convert to rgb
    loss = (loss * 255).astype(np.uint8)
    loss_rgb = cv2.applyColorMap(loss, cv2.COLORMAP_HOT)
    # BGR to RGB
    loss_rgb = cv2.cvtColor(loss_rgb, cv2.COLOR_BGR2RGB)
    return loss_rgb


def depth2color(depth, min, max, rescale=False):
    # normalize depth by min max
    depth = (depth - min) / (max - min)
    depth = depth.clip(0, 1)
    if rescale:
        depth = np.sqrt(depth)
    # convert to rgb
    depth = (depth * 255).astype(np.uint8)
    # depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_HOT)
    depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    # BGR to RGB
    depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)
    return depth_rgb


def save_mesh(mesh, path, bev_size_pixel):
    assert path.endswith(".obj"), "path must be a obj file"
    with torch.no_grad():
        verts = mesh._verts_list[0].detach().cpu()
        faces = mesh._faces_list[0].detach().cpu()
        texture_w, texture_h = bev_size_pixel
        # texture_w, texture_h = 501, 501
        textures = mesh.textures._verts_features_list[0].reshape(texture_w, texture_h, 3).detach().cpu()
        w, h = torch.meshgrid(torch.arange(texture_w, dtype=verts.dtype), torch.arange(texture_h, dtype=verts.dtype))
        verts_uvs = torch.cat((h.unsqueeze(-1)/texture_h, texture_w-1-w.unsqueeze(-1)/texture_w), dim=-1).reshape(-1, 2)
        save_obj(path, verts=verts, faces=faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=textures)


def save_cut_mesh(mesh, path):
    assert path.endswith(".obj"), "path must be a obj file"
    with torch.no_grad():
        verts = mesh._verts_list[0].detach().cpu().numpy()
        faces = mesh._faces_list[0].detach().cpu().numpy()
        vert_colors = mesh.textures._verts_features_list[0][:, :3].detach().cpu().numpy()
        vert_colors = np.concatenate([vert_colors, np.ones((vert_colors.shape[0], 1))], axis=-1)
        m = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces, v_color_matrix=vert_colors)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "vcolor_mesh")
        # save the mesh
        ms.save_current_mesh(path)


def save_cut_label_mesh(mesh, path, color_map):
    assert path.endswith(".obj"), "path must be a obj file"
    with torch.no_grad():
        verts = mesh._verts_list[0].detach().cpu().numpy()
        faces = mesh._faces_list[0].detach().cpu().numpy()
        vert_labels = mesh.textures._verts_features_list[0][:, 3:].detach().cpu().numpy()
        vert_labels = vert_labels.argmax(axis=1)
        vert_colors = np.zeros((vert_labels.shape[0], 3))
        for i in range(vert_labels.shape[0]):
            vert_colors[i] = color_map[vert_labels[i]]
        vert_colors = (vert_colors / 255.0).astype(np.float32)
        vert_colors = np.concatenate([vert_colors, np.ones((vert_colors.shape[0], 1))], axis=-1)
        m = pymeshlab.Mesh(vertex_matrix=verts,face_matrix=faces,v_color_matrix=vert_colors)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "vcolor_mesh")
        # save the mesh
        ms.save_current_mesh(path)


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class Visualizer(nn.Module):
    def __init__(self, device, configs):
        super().__init__()
        self.device = device
        self.configs = configs

        image_size = (self.configs["bev_x_pixel"], self.configs["bev_y_pixel"])
        rotation = torch.from_numpy(np.asarray([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ], dtype=np.float32))[None]
        cx = self.configs["bev_x_length"] / 2
        cy = self.configs["bev_y_length"] / 2
        translation = torch.from_numpy(np.asarray([-cx, -cy, 1], dtype=np.float32))[None]
        image_size_tensor = torch.from_numpy(np.asarray(image_size, dtype=np.float32))[None]
        focal_length = torch.from_numpy(np.asarray([1/cx, 1/cy], dtype=np.float32))[None]
        camera = OrthographicCameras(
            focal_length=focal_length,
            R=rotation,
            T=translation,
            image_size=image_size_tensor,
            device=device
        )

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            ),
            shader=SimpleShader()
        )

    def forward(self, mesh):
        image, depth = self.mesh_renderer(mesh)
        return image, depth
