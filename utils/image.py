import cv2
import numpy as np

def undistort_nearest(cv_image, k, d):
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (cv_image.shape[1], cv_image.shape[0]), cv2.CV_32FC1)
    cv_image_undistorted = cv2.remap(cv_image, mapx, mapy, cv2.INTER_NEAREST)
    return cv_image_undistorted

def render_semantic(label, colors):
    label_bgr = cv2.cvtColor(label.astype("uint8"), cv2.COLOR_GRAY2BGR)
    rendered_label = np.array(cv2.LUT(label_bgr, colors))
    return rendered_label