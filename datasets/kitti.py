import numpy as np
import cv2
from multiprocessing.pool import ThreadPool as Pool
from os import listdir
from os.path import join, isfile
from copy import deepcopy
from datasets.base import BaseDataset
from utils.plane_fit import robust_estimate_flatplane


class KittiDataset(BaseDataset):
    def __init__(self, configs):
        super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        self.sequence = configs["sequence"]
        camera_names = configs["camera_names"]  # image_2 or image_3
        self.choose_pt = [configs["choose_point"]["x"], configs["choose_point"]["y"]]
        x_offset = -configs["center_point"]["x"] + configs["bev_x_length"]/2
        y_offset = -configs["center_point"]["y"] + configs["bev_y_length"]/2
        self.intrinsic = np.asarray([
            [7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
            [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]
        ], dtype=np.float32)  # image_2
        self.transform_02 = np.asarray([[9.999758e-01, -5.267463e-03, -4.552439e-03, 5.956621e-02],
                                        [5.251945e-03, 9.999804e-01, -3.413835e-03, 2.900141e-04],
                                        [4.570332e-03, 3.389843e-03, 9.999838e-01, 2.577209e-03],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.transform_03 = np.asarray([[9.995599e-01, 1.699522e-02, -2.431313e-02, -4.731050e-01],
                                        [-1.704422e-02, 9.998531e-01, -1.809756e-03, 5.551470e-03],
                                        [2.427880e-02, 2.223358e-03, 9.997028e-01, -5.250882e-03],
                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.transform_32 = np.linalg.inv(self.transform_03) @ self.transform_02
        self.extrinsic = np.eye(4)  # only support image_2 now
        self.camera_extrinsics = []
        self.d = 0
        self.camera_height = 1.6
        self.camera2chassis = np.asarray([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.world2bev = np.asarray([
            [1, 0, 0, x_offset],
            [0, 1, 0, y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.min_distance = configs["min_distance"]

        # start loading all filename and poses
        ref_poses = self.get_ref_poses(self.sequence)  # refrece image_2 to chassis

        for camera_idx, camera_name in enumerate(camera_names):
            camera_paths = listdir(join(self.base_dir, "sequences", self.sequence, camera_name))
            camera_paths.sort(key=lambda x: int(x[:-4]))
            last_xy_location = np.asarray([0, 0], dtype=np.float32)
            camera_extrinsic = self.get_extrinsic(camera_name, camera_names[0])
            self.camera_extrinsics.append(camera_extrinsic.astype(np.float32))

            for i, camera_path in enumerate(camera_paths):
                # 1. ref_camera2world
                rel_camera_path = join("sequences", self.sequence, camera_name, camera_path)
                ref_camera2world = ref_poses[i].astype(np.float32)
                xy_location = ref_camera2world[:2, 3]
                center_location = np.asarray([configs["choose_point"]["x"], configs["choose_point"]["y"]])
                center_distance = np.abs(xy_location - center_location)
                if center_distance[0] > configs["bev_x_length"]/2 + 10 or center_distance[1] > configs["bev_y_length"]/2 + 10:
                    continue
                frame_distance = np.sum(np.abs(last_xy_location - xy_location))
                if frame_distance < self.min_distance:
                    continue
                last_xy_location = xy_location

                self.ref_camera2world_all.append(ref_camera2world)

                # 3. label_path
                rel_label_path = rel_camera_path.replace("sequences", "seg_sequences")
                rel_label_path = rel_label_path.replace(".jpg", ".png")
                self.label_filenames_all.append(rel_label_path)

                # 3. camera_path
                self.image_filenames_all.append(rel_camera_path)

                # 4. camera_intrinsic
                self.cameras_K_all.append(self.intrinsic)
                self.cameras_d_all.append(self.d)

                # 5. camera index
                self.cameras_idx_all.append(camera_idx)

        # 6. estimate flat plane
        self.file_check()
        self.label_valid_check()
        ref_camera2world_all = np.array(self.ref_camera2world_all)
        print("before plane estimation, z std = ", ref_camera2world_all[:, 2, 3].std())
        transform_normal2origin = robust_estimate_flatplane(np.array(ref_camera2world_all)[:, :3, 3]).astype(np.float32)
        transform_normal2origin[0, 3] = -self.choose_pt[0]
        transform_normal2origin[1, 3] = -self.choose_pt[1]
        transform_normal2origin[2, 3] += self.camera_height
        self.ref_camera2world_all = transform_normal2origin[None] @ self.ref_camera2world_all
        print("after plane estimation, z std = ", self.ref_camera2world_all[:, 2, 3].std())

        # 7. filter poses in bev range
        all_camera_xy = np.asarray(self.ref_camera2world_all)[:, :2, 3]
        available_mask_x = abs(all_camera_xy[:, 0]) < configs["bev_x_length"] // 2 + 10
        available_mask_y = abs(all_camera_xy[:, 1]) < configs["bev_y_length"] // 2 + 10
        available_mask = available_mask_x & available_mask_y
        available_idx = list(np.where(available_mask)[0])
        print(f"before poses filtering, pose num = {available_mask.shape[0]}")
        self.filter_by_index(available_idx)
        print(f"after poses filtering, pose num = {available_mask.sum()}")

    def loadarray_kitti(self, array):
        input_pose = array
        assert(input_pose.shape[1] == 12)
        length = input_pose.shape[0]
        input_pose = input_pose.reshape(-1, 3, 4)
        bottom = np.zeros((length, 1, 4))
        bottom[:, :, -1] = 1
        transforms = np.concatenate((input_pose, bottom), axis=1)
        return transforms

    def get_ref_poses(self, sequence):
        # odometry_path = join(self.base_dir, f"sequences/gt_pose/{sequence}.txt")
        odometry_path = join(self.base_dir, f"sequences/orbslam2/{sequence}.txt")
        if not isfile(odometry_path):
            return None
        camera_poses = self.loadarray_kitti(np.loadtxt(odometry_path))
        camera_poses = self.camera2chassis @ camera_poses  # refrece to chassis
        return camera_poses

    def get_extrinsic(self, camera_name, camera_name_0):
        assert camera_name_0 == "image_2", "camera_name_0 should be image_2"
        assert camera_name in ["image_2", "image_3"], "camera_name should be image_2 or image_3"
        if camera_name == camera_name_0:
            return np.eye(4)
        else:
            if camera_name_0 == "image_2":
                return self.transform_32
            else:
                return np.linalg.inv(self.extrinsic)

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

    def filter_by_index(self, index):
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.ref_camera2world_all = [self.ref_camera2world_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]

    def set_waypoint(self, center_xy, radius):
        center_xy = np.asarray([center_xy[0], center_xy[1]], dtype=np.float32)
        all_camera_xy = np.asarray(self.ref_camera2world_all)[:, :2, 3]
        distances = np.linalg.norm(all_camera_xy - center_xy, ord=np.inf, axis=1)
        activated_idx = list(np.where(distances < radius)[0])
        self.image_filenames = [self.image_filenames_all[i] for i in activated_idx]
        self.label_filenames = [self.label_filenames_all[i] for i in activated_idx]
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

        # label_off_road = label_off_road | label_movable
        label_off_road = label_movable
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
        input_image = cv2.imread(join(self.base_dir, image_path))
        crop_cy = int(self.resized_image_size[1] * 0.4)
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
        colors *= 4          # background
        colors[7, :] = 1     # Lane marking
        colors[8, :] = 1
        colors[14, :] = 1
        colors[23, :] = 1
        colors[24, :] = 1
        colors[2, :] = 2     # curb
        colors[9, :] = 2     # curb cut
        colors[41, :] = 3    # Manhole
        colors[13, :] = 3    # road
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
        return 5

    @ property
    def filted_color_map(self):
        colors = np.zeros((256, 1, 3), dtype='uint8')
        colors[0, :, :] = [0, 0, 0]         # mask
        colors[1, :, :] = [0, 0, 255]       # all lane
        colors[2, :, :] = [255, 0, 0]       # curb
        colors[3, :, :] = [211, 211, 211]   # road and manhole
        colors[4, :, :] = [157, 234, 50]    # background
        return colors
