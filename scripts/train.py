import argparse
import yaml
from tqdm import tqdm
import random

import numpy as np
import torch
import os
import cv2
from os.path import join
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from pytorch3d.renderer import PerspectiveCameras
from models.loss import L1MaskedLoss, CELossWithMask
from utils.geometry import fps_by_distance
from utils.renderer import Renderer
from utils.visualizer import Visualizer, loss2color, depth2color, save_cut_mesh, save_cut_label_mesh
from utils.wandb_loggers import WandbLogger
from utils.image import render_semantic
from models.pose_model import ExtrinsicModel
from pytorch3d.loss import mesh_laplacian_smoothing
from eval import eval


def set_randomness(args):
    random.seed(args["rand_seed"])
    np.random.seed(args["rand_seed"])
    torch.manual_seed(args["rand_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs():
    parser = argparse.ArgumentParser(description='G4M config')
    parser.add_argument(
        '--config',
        default="configs/local_carla.yaml",
        help='config yaml path')
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    return configs


def train(configs):
    set_randomness(configs)
    if configs["cluster"]:
        os.environ['WANDB_MODE'] = 'offline'
    device = torch.device("cuda:0")

    if configs["dataset"] == "NuscDataset":
        from datasets.nusc import NuscDataset as Dataset
    elif configs["dataset"] == "KittiDataset":
        from datasets.kitti import KittiDataset as Dataset
    else:
        raise NotImplementedError("Dataset not implemented")

    logger = WandbLogger(configs)
    visualizer = Visualizer(device, configs)
    renderer = Renderer().to(device)
    dataset = Dataset(configs)
    supervise_depth_list = ["FsdDataset", "CarlaDataset"]
    # supervise_depth_list = ["CarlaDataset"]

    pose_xy = np.array(dataset.ref_camera2world_all)[:, :2, 3]
    offset_pose_xy = pose_xy - np.asarray([configs["center_point"]["x"], configs["center_point"]["y"]])
    print(f"Get {len(dataset.ref_camera2world_all)} images for mapping")

    # Load grid and optimization toggles
    optim_dict = dict()
    for optim_option in ["vertices_rgb", "vertices_label", "vertices_z", "rotations", "translations"]:
        if configs["lr"].get(optim_option, 0) != 0:
            print("{} optimization is ON".format(optim_option))
            optim_dict[optim_option] = True
        else:
            optim_dict[optim_option] = False
            print("{} optimization is OFF".format(optim_option))

    # Choose Different grid generator according to configs
    if optim_dict["vertices_rgb"] and optim_dict["vertices_label"] and (not optim_dict["vertices_z"]):
        from models.voxel import SquareFlatGridRGBLabel as SquareFlatGrid
    elif optim_dict["vertices_rgb"] and (not optim_dict["vertices_label"]) and optim_dict["vertices_z"]:
        from models.voxel import SquareFlatGridRGBZ as SquareFlatGrid
    elif optim_dict["vertices_rgb"] and (not optim_dict["vertices_label"]) and (not optim_dict["vertices_z"]):
        from models.voxel import SquareFlatGridRGB as SquareFlatGrid
    elif (not optim_dict["vertices_rgb"]) and optim_dict["vertices_label"] and (not optim_dict["vertices_z"]):
        from models.voxel import SquareFlatGridLabel as SquareFlatGrid
    elif (not optim_dict["vertices_rgb"]) and optim_dict["vertices_label"] and optim_dict["vertices_z"]:
        from models.voxel import SquareFlatGridLabelZ as SquareFlatGrid
    elif optim_dict["vertices_rgb"] and optim_dict["vertices_label"] and optim_dict["vertices_z"]:
        from models.voxel import SquareFlatGridRGBLabelZ as SquareFlatGrid
    else:
        raise NotImplementedError("No such grid generator, please check your config[\"lr\"]")

    if optim_dict["vertices_z"]:
        grid = SquareFlatGrid(configs["bev_x_length"], configs["bev_y_length"], offset_pose_xy,
                              configs["bev_resolution"], dataset.num_class, configs["pos_enc"], configs["cut_range"])
    else:
        grid = SquareFlatGrid(configs["bev_x_length"], configs["bev_y_length"], offset_pose_xy,
                              configs["bev_resolution"], dataset.num_class, configs["cut_range"])
    grid = grid.to(device)
    grid.init_vertices_z()

    # Prepare trainable parameters
    parameters = []
    z_parameters = []
    pose_parameters = []
    for param_key, param in grid.named_parameters():
        if "vertices_rgb" in param_key or "vertices_label" in param_key:
            parameters.append({"params": param, "lr": float(configs["lr"][param_key.split('.')[-1]])})
        else:
            z_parameters.append({"params": param, "lr": float(configs["lr"]["vertices_z"])})

    poses = ExtrinsicModel(configs, optim_dict["rotations"], optim_dict["translations"], num_camera=len(dataset.camera_extrinsics)).to(device)
    for param_key, param in poses.named_parameters():
        pose_parameters.append({"params": param, "lr": float(configs["lr"][param_key])})

    # Prepare loss function and optimizer
    optimizer = torch.optim.Adam(parameters)
    scheduler = MultiStepLR(optimizer, milestones=configs["lr_milestones"], gamma=configs["lr_gamma"])
    if optim_dict["vertices_z"]:
        z_optimizer = torch.optim.Adam(z_parameters)
    if optim_dict["translations"] or optim_dict["rotations"]:
        pose_optimizer = torch.optim.Adam(pose_parameters)
    loss_fuction = L1MaskedLoss()
    depth_loss_fuction = L1MaskedLoss()
    CE_loss_with_mask = CELossWithMask()

    radius = configs["waypoint_radius"]
    # Start optimization
    loop = tqdm(range(1, configs["epochs"]+1))
    for epoch in loop:
        waypoints = fps_by_distance(pose_xy, min_distance=radius*2, return_idx=False)
        print(f"epoch-{epoch}: get {waypoints.shape[0]} waypoints")
        loss_dict = dict()
        if optim_dict["vertices_rgb"]:
            loss_dict["render_loss"] = 0
        if optim_dict["vertices_label"]:
            loss_dict["seg_loss"] = 0
        if optim_dict["vertices_z"]:
            loss_dict["laplacian_loss"] = 0
            if configs["dataset"] in supervise_depth_list:
                loss_dict["depth_loss"] = 0
        loss_dict["total_loss"] = 0

        num_frames = 0
        for waypoint in waypoints:
            vertice_waypoint = waypoint + dataset.world2bev[:2, 3]
            if optim_dict["vertices_z"]:
                activation_idx = grid.get_activation_idx(vertice_waypoint, radius)
            dataset.set_waypoint(waypoint, radius * 1.1)
            num_frames += len(dataset)
            print(f"+ {len(dataset)}, num_frames = {num_frames}")
            dataloader = DataLoader(dataset, batch_size=configs["batch_size"],
                                    num_workers=configs["num_workers"],
                                    shuffle=True,
                                    drop_last=True)

            for sample in dataloader:
                for key, ipt in sample.items():
                    if key != "image_path":
                        sample[key] = ipt.clone().detach().to(device)
                if optim_dict["vertices_z"]:
                    mesh = grid(activation_idx, configs["batch_size"])
                else:
                    mesh = grid(configs["batch_size"])
                pose = poses(sample["camera_idx"])
                if epoch >= configs["extrinsic"]["start_epoch"]:
                    transform = pose @ sample["Transform_pytorch3d"]
                else:
                    transform = sample["Transform_pytorch3d"]

                R_pytorch3d = transform[:, :3, :3]
                T_pytorch3d = transform[:, :3, 3]
                focal_pytorch3d = sample["focal_pytorch3d"]
                p0_pytorch3d = sample["p0_pytorch3d"]
                image_shape = sample["image_shape"]
                cameras = PerspectiveCameras(
                    R=R_pytorch3d,
                    T=T_pytorch3d,
                    focal_length=focal_pytorch3d,
                    principal_point=p0_pytorch3d,
                    image_size=image_shape,
                    device=device
                )
                gt_image = sample["image"]
                if optim_dict["vertices_z"] and (configs["dataset"] in supervise_depth_list):
                    gt_depth = sample["depth"]
                gt_seg = sample["static_label"]

                images_feature, depth = renderer({"mesh": mesh, "cameras": cameras})
                silhouette = images_feature[:, :, :, -1]
                silhouette[silhouette > 0] = 1
                silhouette = torch.unsqueeze(silhouette, -1)
                mask = silhouette
                if "static_mask" in sample:
                    static_mask = torch.unsqueeze(sample["static_mask"], -1)
                    mask *= static_mask

                images = images_feature[:, :, :, :3]
                if optim_dict["vertices_rgb"]:
                    images_seg = images_feature[:, :, :, 3:-1]
                else:
                    images_seg = images_feature[:, :, :, :-1]

                optimizer.zero_grad()
                if optim_dict["vertices_z"]:
                    z_optimizer.zero_grad()
                if optim_dict["translations"] or optim_dict["rotations"]:
                    pose_optimizer.zero_grad()
                total_loss = 0
                if optim_dict["vertices_rgb"]:
                    render_loss = loss_fuction(images, gt_image, mask)
                    total_loss += render_loss.mean()
                if optim_dict["vertices_label"]:
                    seg_loss = CE_loss_with_mask(images_seg.reshape(-1, images_seg.shape[-1]),
                                                 gt_seg.reshape(-1), mask.reshape(-1)) * configs["seg_loss_weight"]
                    total_loss += seg_loss
                if optim_dict["vertices_z"]:
                    if configs["dataset"] in supervise_depth_list:
                        mask_depth = gt_depth > 0
                        depth_loss = depth_loss_fuction(depth, gt_depth, mask * mask_depth) * configs["depth_loss_weight"]
                        total_loss += depth_loss.mean()
                    laplacian_loss = mesh_laplacian_smoothing(mesh) * configs["laplacian_loss_weight"]
                    total_loss += laplacian_loss

                total_loss.backward()
                optimizer.step()
                z_optimizer.step() if z_parameters else None
                pose_optimizer.step() if pose_parameters else None
                if optim_dict["vertices_rgb"]:
                    loss_dict["render_loss"] += render_loss.mean().detach().cpu().numpy()
                if optim_dict["vertices_label"]:
                    loss_dict["seg_loss"] += seg_loss.detach().cpu().numpy()
                if optim_dict["vertices_z"]:
                    loss_dict["laplacian_loss"] += laplacian_loss.detach().cpu().numpy()
                    if configs["dataset"] in supervise_depth_list:
                        loss_dict["depth_loss"] += depth_loss.mean().detach().cpu().numpy()

                loss_dict["total_loss"] += total_loss.detach().cpu().numpy()
        scheduler.step()
        with torch.no_grad():
            if optim_dict["vertices_z"]:
                mesh = grid(None, configs["batch_size"])
            else:
                mesh = grid(configs["batch_size"])

        if not configs["cluster"]:
            # Log to wandb
            for key, value in loss_dict.items():
                loss_dict[key] = value / len(dataloader)
            logger.log(loss_dict, epoch)
            bev_features, bev_depth = visualizer(mesh[0])
            if optim_dict["vertices_rgb"]:
                bev_seg = bev_features[0, :, :, 3:-1].detach().cpu().numpy()
            else:
                bev_seg = bev_features[0, :, :, :-1].detach().cpu().numpy()
            gt_image_0 = gt_image[0].detach().cpu().numpy()
            gt_image_0 = (gt_image_0 * 255).astype(np.uint8)
            if optim_dict["vertices_rgb"]:
                render_loss = render_loss[0].detach().cpu().numpy()
                vis_render_loss = loss2color(render_loss)
                logger.log_image("vis_loss", vis_render_loss, epoch)
                bev_rgb = bev_features[0, :, :, :3].detach().cpu().numpy()
                bev_rgb = np.clip(bev_rgb, 0, 1)  # see https://github.com/wandb/client/issues/2722
                bev_rgb = bev_rgb[::-1, ::-1, :]
                render_image = np.clip(images[0].detach().cpu().numpy(), 0, 1)
                logger.log_image("bev_rgb", bev_rgb, epoch)
                logger.log_image("render_image", render_image, epoch)
                logger.log_image("gt_image", gt_image_0, epoch)
            if optim_dict["vertices_label"]:
                bev_seg = np.argmax(bev_seg, axis=-1)
                bev_seg = render_semantic(bev_seg, dataset.filted_color_map)  # RGB fomat
                bev_seg = bev_seg[::-1, ::-1, :]
                render_seg = images_seg[0].detach().cpu().numpy()
                render_seg = np.argmax(render_seg, axis=-1)
                render_seg = render_semantic(render_seg, dataset.filted_color_map)  # RGB fomat
                render_gt_seg = render_semantic(gt_seg[0].detach().cpu().numpy(), dataset.filted_color_map)
                render_mask = (mask[0].detach().cpu().numpy() * 255).astype(np.uint8)
                blend_image = cv2.addWeighted(gt_image_0, 0.5, render_seg, 0.5, 0)
                logger.log_image("render_mask", render_mask, epoch)
                logger.log_image("bev_seg", bev_seg, epoch)
                logger.log_image("render_seg", render_seg, epoch)
                logger.log_image("render_gt_seg", render_gt_seg, epoch)
                logger.log_image("blend_image", blend_image, epoch)

            if optim_dict["vertices_z"]:
                vis_bev_depth = depth2color(bev_depth[0, :, :, 0].detach().cpu().numpy(), min=0.8, max=1.2)
                vis_render_depth = depth2color(depth[0, :, :, 0].detach().cpu().numpy(), min=0, max=100, rescale=True)
                vis_bev_depth = vis_bev_depth[::-1, ::-1, :]
                logger.log_image("vis_bev_depth", vis_bev_depth, epoch)
                logger.log_image("vis_render_depth", vis_render_depth, epoch)
                if configs["dataset"] in supervise_depth_list:
                    kernel = np.ones((10, 10), np.uint8)
                    vis_gt_depth = depth2color(gt_depth[0, :, :, 0].detach().cpu().numpy(), min=0, max=100, rescale=True)
                    vis_gt_depth = cv2.dilate(vis_gt_depth, kernel, iterations=1)
                    logger.log_image("vis_gt_depth", vis_gt_depth, epoch)

                    vis_depth_loss = depth2color(depth_loss[0].detach().cpu().numpy(), min=0, max=20, rescale=False)
                    kernel = np.ones((10, 10), np.uint8)
                    vis_depth_loss = cv2.dilate(vis_depth_loss, kernel, iterations=1)
                    depth_loss_mask = vis_depth_loss.sum(axis=-1) > 0
                    rgb = (gt_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    rgb[depth_loss_mask] = vis_depth_loss[depth_loss_mask]
                    logger.log_image("vis_depth_loss", rgb, epoch)

            description = "Epoch: {}, loss = {:.2}".format(
                epoch,
                loss_dict["total_loss"])
            loop.set_description(description)

    # Save .obj file
    save_cut_mesh(mesh[0], join(logger.dir, f"bev_mesh_epoch_{epoch}.obj"))
    save_cut_label_mesh(mesh[0], join(logger.dir, f"bev_label_mesh_epoch_{epoch}.obj"), dataset.filted_color_map)

    # Save model
    grid.eval()
    poses.eval()
    torch.save(grid, join(logger.dir, "grid_baseline.pt"))
    torch.save(poses, join(logger.dir, "pose_baseline.pt"))
    print(f"Saved model to {logger.dir}")

    if configs["eval"]:
        eval(grid, poses, dataset, renderer, configs, device)


if __name__ == "__main__":
    configs = get_configs()
    train(configs)
