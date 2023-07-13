import torch
from torch import nn
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)


def hard_channel_blend(
    colors: torch.Tensor, fragments,
) -> torch.Tensor:
    """
    Naive blending of top K faces to return an C+1 image
    Args:
        colors: (N, H, W, K, C) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
    Returns:
        RGBA pixel_channels: (N, H, W, C+1)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Mask for the background.
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    background_color = torch.ones(colors.shape[-1], dtype=colors.dtype, device=colors.device)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, C)

    # Concat with the alpha channel.
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, C+1)


class SimpleShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        images = hard_channel_blend(texels, fragments)
        return images


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.raster_settings = None

    def set_rasterization(self, cameras):
        image_size = tuple(cameras.image_size[0].detach().cpu().numpy())
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    def forward(self, input):
        mesh = input["mesh"]
        cameras = input["cameras"]
        if self.raster_settings is None:
            self.set_rasterization(cameras)

        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        images, depth = mesh_renderer(mesh)
        return images, depth


class RendererBev(nn.Module):
    def __init__(self):
        super().__init__()
        self.raster_settings = None
        self.image_size = tuple((640, 1024))  # FOV cameras do not have image_size

    def set_rasterization(self):
        image_size = self.image_size
        self.raster_settings = RasterizationSettings(
            image_size=(int(image_size[0]), int(image_size[1])),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    def forward(self, input):
        mesh = input["mesh"]
        cameras = input["cameras"]
        if self.raster_settings is None:
            self.set_rasterization()

        mesh_renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SimpleShader()
        )
        images, depth = mesh_renderer(mesh)
        return images, depth
