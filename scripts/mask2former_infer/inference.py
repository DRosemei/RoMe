# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
from kitti_dataset import CrawlKittiDataPath
from nuscenes_scenes import crawl_scenes_paths
from nuscenes_dataset import CrawlNuScenesDataPath
from predictor import VisualizationDemo
from mask2former import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from os.path import join
from pathlib import Path
import tqdm
import numpy as np
import cv2
import warnings
import time
import tempfile
import argparse
import glob
import multiprocessing as mp
import os
from poplib import CR

# fmt: off
import sys
from turtle import pd
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--base_dir",
        default="#####/Nuscenes/sweeps/",  # "samples" contain key frames
        help="nuScenes base dir",
    )

    parser.add_argument(
        "--save_dir",
        default="#####/KittiOdom/sequences",
        help="nuScenes base dir",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    # camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    #                 "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    # camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    # camera_names = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    # # create output dir
    # for cam in camera_names:
    #     cam_path = join(args.save_dir, cam)
    #     Path(cam_path).mkdir(parents=True, exist_ok=True)
    #     cam_path = cam_path.replace("/CAM", "/seg_CAM")
    #     Path(cam_path).mkdir(parents=True, exist_ok=True)
    #     cam_path = cam_path.replace("/seg_CAM", "/vis_seg_CAM")
    #     Path(cam_path).mkdir(parents=True, exist_ok=True)
    # # Nuscenes
    # file_paths = CrawlNuScenesDataPath(args.base_dir, camera_names)

    # root_dir = "#####/Nuscenes"
    # version = "mini"
    # scenes = ["scene-0655"]
    # camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    #                 "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    # file_paths = crawl_scenes_paths(root_dir, version, scenes, camera_names)

    # save_paths = [file_path.replace(args.base_dir, args.save_dir)  for file_path in file_paths]

    # vis_label_save_paths = [save_path.replace("/CAM", "/vis_seg_CAM")  for save_path in save_paths]

    # label_save_paths = [save_path.replace("/CAM", "/seg_CAM")  for save_path in save_paths]
    # label_save_paths = [label_save_path.replace(".jpg", ".png")  for label_save_path in label_save_paths]

    # Kitti
    base_dir = "#####/KittiOdom/sequences"
    save_dir = "#####/KittiOdom/seg_sequences"
    vis_dir = "#####/KittiOdom/vis_seg_sequences"
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    for sequence in sequences:
        sequence_path = join(save_dir, sequence, "image_3")
        Path(sequence_path).mkdir(parents=True, exist_ok=True)

    file_paths = CrawlKittiDataPath(base_dir, sequences, "image_3")
    label_save_paths = [file_path.replace(base_dir, save_dir) for file_path in file_paths]
    vis_label_save_paths = [file_path.replace(base_dir, vis_dir) for file_path in file_paths]
    for i in tqdm.tqdm(range(len(file_paths))):
        # use PIL, to be consistent with evaluation
        source_name = file_paths[i]
        # vis_label_name = vis_label_save_paths[i]
        label_name = label_save_paths[i]
        img = read_image(source_name, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        # visualized_output.save(vis_label_name)
        save_img = predictions["sem_seg"].argmax(dim=0).cpu().numpy()  # (H, W)
        cv2.imwrite(label_name, save_img)
