import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes


def main(configs):
    nusc = NuScenes(version="v1.0-{}".format(configs.version), dataroot=configs.dataroot, verbose=True)
    print(f"Selecting scenes in {configs.location} with center {configs.xy_center} and radius {configs.xy_radius}")
    xy_center = np.array(configs.xy_center)[None]
    radius = configs.xy_radius
    samples = [samp for samp in nusc.sample]
    scene_names = []
    for scene in nusc.scene:
        if configs.filter_rain and configs.filter_night:
            if "Rain" in scene["description"] or "Night" in scene["description"]:
                continue
        scene_name = scene["name"]
        location = nusc.get("log", scene["log_token"])["location"]
        if location != configs.location:
            continue
        records = [samp for samp in samples if
                   nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        # sort by timestamp (only to make chronological viz easier)
        records.sort(key=lambda x: (x["timestamp"]))
        xy_scene = []
        for index in range(len(records)):
            rec = records[index]
            rec_token = rec["data"]["CAM_FRONT"]
            samp = nusc.get("sample_data", rec_token)
            pose_chassis2global = nusc.get("ego_pose", samp["ego_pose_token"])
            xy_scene.append(pose_chassis2global["translation"][:2])
        xy_scene = np.array(xy_scene)
        diff = np.linalg.norm(xy_scene - xy_center, axis=1, ord=2)
        # print(f"{scene_name} {diff.min()} {diff.max()}")
        if diff.min() < radius:
            scene_names.append(scene_name)
    print(f"Found {len(scene_names)} scenes")
    print(f"{scene_names}")
    return scene_names


def get_args():
    parser = argparse.ArgumentParser(description="Get NuScenes scenes by center and radius", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version",                type=str,           default="trainval",         help="NuScenes version: trainval or  mini")
    parser.add_argument("--dataroot",               type=str,           default="/mnt/data/Nuscenes/Nuscenes/", help="NuScenes dataroot")

    parser.add_argument("--location",               type=str,           default="boston-seaport",   help="location")
    parser.add_argument("--xy_center",              type=list,          default=[820, 516],       help="xy_center")
    parser.add_argument("--xy_radius",              type=float,         default=20,                 help="xy_radius")
    parser.add_argument("--filter_rain",            type=bool,          default=True,               help="filter rain scenes")
    parser.add_argument("--filter_night",           type=bool,          default=True,               help="filter night scenes")
    return parser.parse_args()


if __name__ == "__main__":
    # locations "boston-seaport", "boston-seaport", "singapore-queensto", "singapore-hollandv"
    configs = get_args()
    scene_names = main(configs)
