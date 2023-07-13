import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
from models.voxel import clean_nan


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


class PoseModel(nn.Module):
    def __init__(self, optim_rotation=False, optim_translation=False, num_frame=None):
        super().__init__()
        if optim_rotation:
            self.rotations = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3) axis angle
        else:
            rotations = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            self.register_buffer("rotations", rotations)
        if optim_translation:
            self.translations = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3)
        else:
            translations = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            self.register_buffer("translations", translations)

    def forward(self, idx):
        if idx is not None:
            translations = self.translations[idx]
            rotations = self.rotations[idx]
        else:
            translations = self.translations
            rotations = self.rotations
        rots = axis_angle_to_matrix(1./180.*np.pi * torch.tanh(rotations))
        translations = 0.2*torch.tanh(translations.unsqueeze(2))
        poses = convert3x4_4x4(torch.cat((rots, translations), dim=2))
        if self.translations.requires_grad:
            self.translations.register_hook(clean_nan)
        if self.rotations.requires_grad:
            self.rotations.register_hook(clean_nan)
        return poses


class ExtrinsicModel(nn.Module):
    def __init__(self, configs, optim_rotation=False, optim_translation=False, num_camera=None):
        super().__init__()
        self.configs = configs
        if optim_rotation:
            self.rotations = nn.Parameter(torch.zeros(size=(num_camera, 3), dtype=torch.float32))  # (N, 3) axis angle
        else:
            rotations = torch.zeros(size=(num_camera, 3), dtype=torch.float32)
            self.register_buffer("rotations", rotations)
        if optim_translation:
            self.translations = nn.Parameter(torch.zeros(size=(num_camera, 3), dtype=torch.float32))  # (N, 3)
        else:
            translations = torch.zeros(size=(num_camera, 3), dtype=torch.float32)
            self.register_buffer("translations", translations)

    def forward(self, camera_idx):
        translations = self.translations[camera_idx]
        rotations = self.rotations[camera_idx]
        rots = axis_angle_to_matrix(self.configs["extrinsic"]["rotation_deg"]/180.*np.pi * torch.tanh(rotations))
        translations = self.configs["extrinsic"]["translation_m"]*torch.tanh(translations.unsqueeze(2))
        poses = convert3x4_4x4(torch.cat((rots, translations), dim=2))
        if self.translations.requires_grad:
            self.translations.register_hook(clean_nan)
        if self.rotations.requires_grad:
            self.rotations.register_hook(clean_nan)
        return poses


class PoseModelv3(nn.Module):
    def __init__(self, optim_rotation=False, optim_translation=False, num_frame=None, num_camera=None):
        super().__init__()
        if optim_rotation:
            self.rotations_ref = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3) axis angle
            self.rotations_cam = nn.Parameter(torch.zeros(size=(num_camera-1, 3), dtype=torch.float32))  # (N, 3) axis angle
        else:
            rotations_ref = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            rotations_cam = torch.zeros(size=(num_camera-1, 3), dtype=torch.float32)
            self.register_buffer("rotations_ref", rotations_ref)
            self.register_buffer("rotations_cam", rotations_cam)
        if optim_translation:
            self.translations_ref = nn.Parameter(torch.zeros(size=(num_frame, 3), dtype=torch.float32))  # (N, 3)
            self.translations_cam = nn.Parameter(torch.zeros(size=(num_camera-1, 3), dtype=torch.float32))  # (N, 3)
        else:
            translations_ref = torch.zeros(size=(num_frame, 3), dtype=torch.float32)
            translations_cam = torch.zeros(size=(num_camera-1, 3), dtype=torch.float32)
            self.register_buffer("translations_ref", translations_ref)
            self.register_buffer("translations_cam", translations_cam)

    def forward(self, frame_idx, camera_idx):
        if frame_idx is not None:
            translations = self.translations[frame_idx]
            rotations = self.rotations[frame_idx]
        else:
            translations = self.translations
            rotations = self.rotations
        rots = axis_angle_to_matrix(0.2/180.*np.pi * torch.tanh(rotations))
        translations = 0.05*torch.tanh(translations.unsqueeze(2))
        poses = convert3x4_4x4(torch.cat((rots, translations), dim=2))
        if self.translations.requires_grad:
            self.translations.register_hook(clean_nan)
        if self.rotations.requires_grad:
            self.rotations.register_hook(clean_nan)
        return poses
