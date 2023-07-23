
from os.path import join
from nuscenes.nuscenes import NuScenes

def crawl_scenes_paths(root_dir, version, scenes, camera_names):
    nusc = NuScenes(version="v1.0-{}".format(version), dataroot=root_dir, verbose=True)
    samples = [samp for samp in nusc.sample]
    paths = []
    for scene in nusc.scene:
        scene_name = scene["name"]
        if scene_name not in scenes:
            continue
        records = [samp for samp in samples if
                    nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        # sort by timestamp (only to make chronological viz easier)
        records.sort(key=lambda x: (x['timestamp']))
        # interpolate images from 2HZ to 12 HZ
        for index in range(len(records)):
            rec = records[index]
            for cam in camera_names:
                # compute camera key frame poses
                rec_token = rec["data"][cam]
                samp = nusc.get("sample_data", rec_token)
                flag = True  
                # compute first key frame and framse between first frame and second frame
                while flag or not samp["is_key_frame"]: 
                    flag = False
                    rel_camera_path = samp["filename"]
                    camera_path = join(root_dir, rel_camera_path)
                    paths.append(camera_path)
                    if samp["next"] != "":
                        samp = nusc.get('sample_data', samp["next"])
                    else:
                        break
    return paths


if __name__ == "__main__":
    # locations "boston-seaport", "boston-seaport", "singapore-queensto", "singapore-hollandv"
    root_dir = "#####/Nuscenes"
    version = "trainval"
    scenes = ["scene-0546", "scene-0556", "scene-0558", "scene-0769"]
    camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    paths = crawl_scenes_paths(root_dir, version, scenes, camera_names)