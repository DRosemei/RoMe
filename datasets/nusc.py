import numpy as np
import cv2
from multiprocessing.pool import ThreadPool as Pool
from os.path import join
from copy import deepcopy
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from skspatial.objects import Plane, Points
from datasets.base import BaseDataset
from utils.plane_fit import robust_estimate_flatplane


class NuscDataset(BaseDataset):
    def __init__(self, configs):
        self.nusc = NuScenes(version="v1.0-{}".format(configs["version"]),
                             dataroot=configs["base_dir"],
                             verbose=True)
        self.version = configs["version"]
        self.replace_name = configs["replace_name"]
        super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        clip_list = configs["clip_list"]
        camera_names = configs["camera_names"]
        x_offset = -configs["center_point"]["x"] + configs["bev_x_length"]/2
        y_offset = -configs["center_point"]["y"] + configs["bev_y_length"]/2
        self.world2bev = np.asarray([
            [1, 0, 0, x_offset],
            [0, 1, 0, y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.min_distance = configs["min_distance"]
        self.camera_extrinsics = []
        # start loading all filename and poses
        samples = [samp for samp in self.nusc.sample]
        lidar_height = []
        lidar2world_all = []
        for scene_name in tqdm(clip_list, desc="Loading data clips"):
            records = [samp for samp in samples if
                       self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
            # sort by timestamp (only to make chronological viz easier)
            records.sort(key=lambda x: (x['timestamp']))

            # interpolate images from 2HZ to 12 HZ
            for index in range(len(records)):
                rec = records[index]
                # comput lidar key frame poses
                rec_token = rec["data"]["LIDAR_TOP"]
                samp = self.nusc.get("sample_data", rec_token)
                lidar2chassis = self.compute_extrinsic2chassis(samp)
                chassis2world = self.compute_chassis2world(samp)
                lidar2world = chassis2world @ lidar2chassis
                lidar2world_all.append(lidar2world)
                lidar_height.append(lidar2chassis[2, 3])
                for camera_idx, cam in enumerate(camera_names):
                    # compute camera key frame poses
                    rec_token = rec["data"][cam]
                    samp = self.nusc.get("sample_data", rec_token)
                    camera2chassis = self.compute_extrinsic2chassis(samp)
                    if cam == "CAM_FRONT":
                        camera_front2_camera_ref = np.eye(4)
                        camera_ref2_camera_front = np.eye(4)
                    else:
                        rec_token_front = rec["data"]["CAM_FRONT"]
                        samp_front = self.nusc.get("sample_data", rec_token_front)
                        camera_front2_camera_ref = self.compute_extrinsic(samp_front, samp)
                        camera_ref2_camera_front = np.linalg.inv(camera_front2_camera_ref)
                    self.camera_extrinsics.append(camera_ref2_camera_front.astype(np.float32))
                    flag = True
                    # compute first key frame and framse between first frame and second frame
                    while flag or not samp["is_key_frame"]:
                        flag = False
                        rel_camera_path = samp["filename"]
                        if True:
                            camera2chassis = self.compute_extrinsic2chassis(samp)
                            # 1. label path
                            rel_label_path = rel_camera_path.replace("/CAM", "/seg_CAM")
                            rel_label_path = rel_label_path.replace(".jpg", ".png")
                            if self.replace_name:
                                rel_label_path = rel_label_path.replace("+", "_")
                            self.label_filenames_all.append(rel_label_path)

                            # 2. camera path
                            self.image_filenames_all.append(rel_camera_path)

                            # 3. camera2world
                            chassis2world = self.compute_chassis2world(samp)

                            ref_camera2world = chassis2world @ camera2chassis @ camera_front2_camera_ref

                            self.ref_camera2world_all.append(ref_camera2world.astype(np.float32))

                            # 4.camera intrinsic
                            calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
                            intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
                            self.cameras_K_all.append(intrinsic.astype(np.float32))

                            # 5. camera index
                            self.cameras_idx_all.append(camera_idx)
                        # not key frames
                        if samp["next"] != "":
                            samp = self.nusc.get('sample_data', samp["next"])
                        else:
                            break

        # 6. estimate flat plane
        self.file_check()
        # self.label_valid_check()

        lidar2world_all = np.array(lidar2world_all)
        print("before plane estimation, z std = ", lidar2world_all[:, 2].std())
        lidar_height = np.array(lidar_height).mean()

        transform_normal2origin = robust_estimate_flatplane(np.array(lidar2world_all)[:, :3, 3]).astype(np.float32)
        transform_normal2origin[2, 3] += lidar_height
        lidar2world_all = transform_normal2origin[None] @ lidar2world_all
        print("after plane estimation, z std = ", lidar2world_all[:, 2].std())
        self.ref_camera2world_all = transform_normal2origin[None] @ np.array(self.ref_camera2world_all)

        # 7. filter poses in bev range
        all_camera_xy = np.asarray(self.ref_camera2world_all)[:, :2, 3]
        available_mask_x = abs(all_camera_xy[:, 0]) < configs["bev_x_length"] // 2 + 10
        available_mask_y = abs(all_camera_xy[:, 1]) < configs["bev_y_length"] // 2 + 10
        available_mask = available_mask_x & available_mask_y
        available_idx = list(np.where(available_mask)[0])
        print(f"before poses filtering, pose num = {available_mask.shape[0]}")
        self.filter_by_index(available_idx)
        print(f"after poses filtering, pose num = {available_mask.sum()}")

    def compute_chassis2world(self, samp):
        """transform sensor in world coordinate"""
        # comput current frame Homogeneous transformation matrix : from chassis 2 global
        pose_chassis2global = self.nusc.get("ego_pose", samp['ego_pose_token'])
        chassis2global = transform_matrix(pose_chassis2global['translation'],
                                          Quaternion(pose_chassis2global['rotation']),
                                          inverse=False)
        return chassis2global

    def compute_extrinsic(self, samp_a, samp_b):
        """transform from sensor_a to sensor_b"""
        sensor_a2chassis = self.compute_extrinsic2chassis(samp_a)
        sensor_b2chassis = self.compute_extrinsic2chassis(samp_b)
        sensor_a2sensor_b = np.linalg.inv(sensor_b2chassis) @ sensor_a2chassis
        return sensor_a2sensor_b

    def compute_extrinsic2chassis(self, samp):
        calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        rot = np.array((Quaternion(calibrated_sensor["rotation"]).rotation_matrix))
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis

    def file_check(self):
        image_paths = [join(self.base_dir, image_path) for image_path in self.image_filenames_all]
        label_paths = [join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        image_exists = np.asarray(self.check_filelist_exist(image_paths))
        label_exists = np.asarray(self.check_filelist_exist(label_paths))
        available_index = list(np.where(image_exists * label_exists)[0])
        print(f"Drop {len(image_paths) - len(available_index)} frames out of {len(image_paths)} by file exists check")
        self.filter_by_index(available_index)

    def label_valid_check(self):
        label_paths = [join(self.image_dir, label_path) for label_path in self.label_filenames_all]
        label_valid = np.asarray(self.check_label_valid(label_paths))
        available_index = list(np.where(label_valid)[0])
        print(f"Drop {len(label_paths) - len(available_index)} frames out of {len(label_paths)} by label valid check")
        self.filter_by_index(available_index)

    def label_valid(self, label_name):
        label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
        label_movable = label >= 52
        ratio_movable = label_movable.sum() / label_movable.size
        label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
            | ((15 <= label) & (label <= 22)) | ((25 <= label) & (label <= 40)) | (label >= 42)
        ratio_static = label_off_road.sum() / label_off_road.size
        if ratio_movable > 0.3 or ratio_static > 0.9:
            return False
        else:
            return True

    def check_label_valid(self, filelist):
        with Pool(32) as p:
            exist_list = p.map(self.label_valid, filelist)
        return exist_list

    def estimate_flatplane(self, lidar2world_all, lidar_height):
        xyz = lidar2world_all[:, :3, 3]
        points = Points(xyz)
        plane = Plane.best_fit(points)
        normal_z = np.asarray(plane.normal)
        origin_x = np.asarray([1, 0, 0])
        normal_y = np.cross(normal_z, origin_x)
        normal_y = normal_y / np.linalg.norm(normal_y)
        normal_x = np.cross(normal_y, normal_z)
        normal_x = normal_x / np.linalg.norm(normal_x)
        rotation_origin2normal = np.asarray([normal_x, normal_y, normal_z]).T
        translation_origin2normal = np.asarray(plane.point)
        transform_origin2normal = np.eye(4)
        transform_origin2normal[:3, :3] = rotation_origin2normal
        transform_origin2normal[:3, 3] = translation_origin2normal
        transform_normal2origin = np.linalg.inv(transform_origin2normal)
        transform_normal2origin[2, 3] += lidar_height
        return transform_normal2origin.astype(np.float32)

    def filter_by_index(self, index):
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.ref_camera2world_all = [self.ref_camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]

    def set_waypoint(self, center_xy, radius):
        center_xy = np.asarray([center_xy[0], center_xy[1]], dtype=np.float32)
        all_camera_xy = np.asarray(self.ref_camera2world_all)[:, :2, 3]
        distances = np.linalg.norm(all_camera_xy - center_xy, ord=np.inf, axis=1)
        activated_idx = list(np.where(distances < radius)[0])
        self.image_filenames = [self.image_filenames_all[i] for i in activated_idx]
        self.label_filenames = [self.label_filenames_all[i] for i in activated_idx]
        self.ref_camera2world = [self.ref_camera2world_all[i] for i in activated_idx]
        self.cameras_idx = [self.cameras_idx_all[i] for i in activated_idx]
        self.cameras_K = [self.cameras_K_all[i] for i in activated_idx]
        self.ref_camera2world = [self.ref_camera2world_all[i] for i in activated_idx]
        self.activated_idx = activated_idx

    def label2mask(self, label):
        # Bird, Ground Animal, Curb, Fence, Guard Rail,
        # Barrier, Wall, Bike Lane, Crosswalk - Plain, Curb Cut,
        # Parking, Pedestrian Area, Rail Track, Road, Service Lane,
        # Sidewalk, Bridge, Building, Tunnel, Person,
        # Bicyclist, Motorcyclist, Other Rider, Lane Marking - Crosswalk, Lane Marking - General,
        # Mountain, Sand, Sky, Snow, Terrain,
        # Vegetation, Water, Banner, Bench, Bike Rack,
        # Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box,
        # Mailbox, Manhole, Phone Booth, Pothole, Street Light,
        # Pole, Traffic Sign Frame, Utility Pole, Traffic Light, Traffic Sign (Back),
        # Traffic Sign (Front), Trash Can, Bicycle, Boat, Bus,
        # Car, Caravan, Motorcycle, On Rails, Other Vehicle,
        # Trailer, Truck, Wheeled Slow, Car Mount, Ego Vehicle
        mask = np.ones_like(label)
        label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
            | ((16 <= label) & (label <= 22)) | ((25 <= label) & (label <= 28)) | ((30 <= label) & (label <= 40)) | (label >= 42)

        # dilate itereation 2 for moving objects
        label_movable = label >= 52
        kernel = np.ones((10, 10), dtype=np.uint8)
        label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(np.bool)

        label_off_road = label_off_road | label_movable
        mask[label_off_road] = 0
        label[~(mask.astype(np.bool))] = 64
        mask = mask.astype(np.float32)
        return mask, label

    def __getitem__(self, idx):
        sample = dict()
        sample["idx"] = idx
        sample["camera_idx"] = self.cameras_idx[idx]
        sample["camera2ref"] = self.camera_extrinsics[sample["camera_idx"]]

        # read image
        image_path = self.image_filenames[idx]
        sample["image_path"] = image_path
        input_image = cv2.imread(join(self.base_dir, image_path))
        camera_name = image_path.split("/")[-2]
        crop_cy = int(self.resized_image_size[1] / 2)
        K = self.cameras_K[idx]
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        sample["image"] = (np.asarray(resized_image)/255.0).astype(np.float32)

        # read label
        label_path = join(self.image_dir, self.label_filenames[idx])
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
        mask, label = self.label2mask(resized_label)
        if camera_name == "CAM_BACK":
            h = mask.shape[0]
            mask[int(0.83 * h):, :] = 0
        label = self.remap_semantic(label).astype(np.long)

        mask = mask[crop_cy:, :]  # crop the sky
        label = label[crop_cy:, :]
        sample["static_mask"] = mask
        sample["static_label"] = label

        cv_camera2world = self.ref_camera2world[idx] @ sample["camera2ref"]  # fsd camera to fsd world
        camera2world = self.world2bev @ cv_camera2world
        sample["world2camera"] = np.linalg.inv(camera2world)
        resized_K = deepcopy(K)
        width_scale = self.resized_image_size[0]/origin_image_size[1]
        height_scale = self.resized_image_size[1]/origin_image_size[0]
        resized_K[0, :] *= width_scale
        resized_K[1, :] *= height_scale
        resized_K[1, 2] -= crop_cy
        sample["camera_K"] = resized_K
        sample["image_shape"] = np.asarray(sample["image"].shape)[:2]
        sample = self.opencv_camera2pytorch3d_(sample)
        return sample

    @ property
    def label_remaps(self):
        colors = np.ones((256, 1), dtype="uint8")
        colors *= 6          # background
        colors[7, :] = 1     # Lane marking
        colors[8, :] = 1
        colors[14, :] = 1
        colors[23, :] = 1
        colors[24, :] = 1
        colors[2, :] = 2     # curb
        colors[9, :] = 2     # curb cut
        colors[41, :] = 3    # Manhole
        colors[13, :] = 3    # road
        colors[15, :] = 4    # sidewalk
        colors[29, :] = 5    # terrain
        return colors

    @ property
    def origin_color_map(self):
        colors = np.zeros((256, 1, 3), dtype='uint8')
        colors[0, :, :] = [165, 42, 42]  # Bird
        colors[1, :, :] = [0, 192, 0]  # Ground Animal
        colors[2, :, :] = [196, 196, 196]  # Curb
        colors[3, :, :] = [190, 153, 153]  # Fence
        colors[4, :, :] = [180, 165, 180]  # Guard Rail
        colors[5, :, :] = [90, 120, 150]  # Barrier
        colors[6, :, :] = [102, 102, 156]  # Wall
        colors[7, :, :] = [128, 64, 255]  # Bike Lane
        colors[8, :, :] = [140, 140, 200]  # Crosswalk - Plain
        colors[9, :, :] = [170, 170, 170]  # Curb Cut
        colors[10, :, :] = [250, 170, 160]  # Parking
        colors[11, :, :] = [96, 96, 96]  # Pedestrian Area
        colors[12, :, :] = [230, 150, 140]  # Rail Track
        colors[13, :, :] = [128, 64, 128]  # Road
        colors[14, :, :] = [110, 110, 110]  # Service Lane
        colors[15, :, :] = [244, 35, 232]  # Sidewalk
        colors[16, :, :] = [150, 100, 100]  # Bridge
        colors[17, :, :] = [70, 70, 70]  # Building
        colors[18, :, :] = [150, 120, 90]  # Tunnel
        colors[19, :, :] = [220, 20, 60]  # Person
        colors[20, :, :] = [255, 0, 0]  # Bicyclist
        colors[21, :, :] = [255, 0, 100]  # Motorcyclist
        colors[22, :, :] = [255, 0, 200]  # Other Rider
        colors[23, :, :] = [200, 128, 128]  # Lane Marking - Crosswalk
        colors[24, :, :] = [255, 255, 255]  # Lane Marking - General
        colors[25, :, :] = [64, 170, 64]  # Mountain
        colors[26, :, :] = [230, 160, 50]  # Sand
        colors[27, :, :] = [70, 130, 180]  # Sky
        colors[28, :, :] = [190, 255, 255]  # Snow
        colors[29, :, :] = [152, 251, 152]  # Terrain
        colors[30, :, :] = [107, 142, 35]  # Vegetation
        colors[31, :, :] = [0, 170, 30]  # Water
        colors[32, :, :] = [255, 255, 128]  # Banner
        colors[33, :, :] = [250, 0, 30]  # Bench
        colors[34, :, :] = [100, 140, 180]  # Bike Rack
        colors[35, :, :] = [220, 220, 220]  # Billboard
        colors[36, :, :] = [220, 128, 128]  # Catch Basin
        colors[37, :, :] = [222, 40, 40]  # CCTV Camera
        colors[38, :, :] = [100, 170, 30]  # Fire Hydrant
        colors[39, :, :] = [40, 40, 40]  # Junction Box
        colors[40, :, :] = [33, 33, 33]  # Mailbox
        colors[41, :, :] = [100, 128, 160]  # Manhole
        colors[42, :, :] = [142, 0, 0]  # Phone Booth
        colors[43, :, :] = [70, 100, 150]  # Pothole
        colors[44, :, :] = [210, 170, 100]  # Street Light
        colors[45, :, :] = [153, 153, 153]  # Pole
        colors[46, :, :] = [128, 128, 128]  # Traffic Sign Frame
        colors[47, :, :] = [0, 0, 80]  # Utility Pole
        colors[48, :, :] = [250, 170, 30]  # Traffic Light
        colors[49, :, :] = [192, 192, 192]  # Traffic Sign (Back)
        colors[50, :, :] = [220, 220, 0]  # Traffic Sign (Front)
        colors[51, :, :] = [140, 140, 20]  # Trash Can
        colors[52, :, :] = [119, 11, 32]  # Bicycle
        colors[53, :, :] = [150, 0, 255]  # Boat
        colors[54, :, :] = [0, 60, 100]  # Bus
        colors[55, :, :] = [0, 0, 142]  # Car
        colors[56, :, :] = [0, 0, 90]  # Caravan
        colors[57, :, :] = [0, 0, 230]  # Motorcycle
        colors[58, :, :] = [0, 80, 100]  # On Rails
        colors[59, :, :] = [128, 64, 64]  # Other Vehicle
        colors[60, :, :] = [0, 0, 110]  # Trailer
        colors[61, :, :] = [0, 0, 70]  # Truck
        colors[62, :, :] = [0, 0, 192]  # Wheeled Slow
        colors[63, :, :] = [32, 32, 32]  # Car Mount
        colors[64, :, :] = [120, 10, 10]  # Ego Vehicle
        # colors[65, :, :] = [0, 0, 0] # Unlabeled
        return colors

    @property
    def num_class(self):
        return 7

    @ property
    def filted_color_map(self):
        colors = np.zeros((256, 1, 3), dtype='uint8')
        colors[0, :, :] = [0, 0, 0]         # mask
        colors[1, :, :] = [0, 0, 255]       # all lane
        colors[2, :, :] = [255, 0, 0]       # curb
        colors[3, :, :] = [211, 211, 211]   # road and manhole
        colors[4, :, :] = [0, 191, 255]     # sidewalk
        colors[5, :, :] = [152, 251, 152]   # terrain
        colors[6, :, :] = [157, 234, 50]    # background
        return colors
