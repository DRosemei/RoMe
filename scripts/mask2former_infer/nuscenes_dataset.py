import os

def CrawlNuScenesDataPath(base_dir, camera_names):
    """get all image_paths under base_dir
    """
    image_paths = []
    for camera_name in camera_names:
        camera_dir = os.path.join(base_dir, camera_name)
        image_rel_list = os.listdir(camera_dir)
        image_abs_list = [os.path.join(base_dir, camera_name, image_rel) for image_rel in image_rel_list]
        image_paths += image_abs_list
        print(image_paths[-1])
    return image_paths

    
if __name__ == "__main__":
    base_dir = "#####/Nuscenes/samples/"
    camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    image_paths = CrawlNuScenesDataPath(base_dir, camera_names)
    # print(image_paths)