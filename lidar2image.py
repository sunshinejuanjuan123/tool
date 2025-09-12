import argparse
import json
import os
import struct

import cv2
import numpy as np
import open3d as o3d
from matplotlib import cm
from PIL import Image


def load_pts(file_path):
    pts = []
    pcd = o3d.geometry.PointCloud()
    FORMAT = "<4fHd"
    POINT_SIZE = struct.calcsize(FORMAT)
    with open(file_path, "rb") as fin:
        idx = 0
        while True:
            data = fin.read(POINT_SIZE)
            if not data:
                break
            x, y, z, intensity, ring, timestamp = struct.unpack(FORMAT, data)
            pcd.points.append([x, y, z])
            pcd.colors.append([intensity / 255, 0, 0])
            idx += 1
        real_num = idx
        print(f"real_num = {real_num}")
    return pcd


def proj_lidar(
    root_path, pcd_path, camera_path, image_filled_key
):
    intrinsic_path = os.path.join(
        root_path, f"calib/center_camera_fov30/center_camera_fov30-intrinsic.json"
    )
    extrinsic_path = os.path.join(
        root_path, f"calib/center_camera_fov30/center_camera_fov30-to-car_center-extrinsic.json"
    )
    image_filled_path = os.path.join(
        root_path,
        f"camera/center_camera_fov30_proj/{image_filled_key}_filled_intensity.jpg",
    )

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    image0 = cv2.imread(camera_path)
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

    intrinsic = json.load(open(intrinsic_path))
    extrinsic = json.load(open(extrinsic_path))
    intrinsic_np = np.array(intrinsic["value0"]["param"]["cam_K_new"]["data"])
    extrinsic_np = np.array(
        extrinsic["center_camera_fov30-to-car_center-extrinsic"]["param"]["sensor_calib"][
            "data"
        ]
    )
    camera2car = extrinsic_np

    lidar2camera = np.linalg.inv(camera2car)
    K = intrinsic_np
    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points_camera_fov30 = (lidar2camera @ points_h.T).T

    pcd_camera_fov30 = o3d.geometry.PointCloud()
    pcd_camera_fov30.points = o3d.utility.Vector3dVector(points_camera_fov30[:, :3])
    o3d.io.write_point_cloud(
        pcd_path.replace(".pcd", "_camera_fov30.ply"), pcd_camera_fov30
    )

    points_camera_fov30 = points_camera_fov30[:, :3] / points_camera_fov30[:, 3:4]
    u = K[0, 0] * points_camera_fov30[:, 0] / points_camera_fov30[:, 2] + K[0, 2]
    v = K[1, 1] * points_camera_fov30[:, 1] / points_camera_fov30[:, 2] + K[1, 2]
    depth = points_camera_fov30[:, 2]

    min_depth = 0
    max_depth = 1000000
    mask = np.logical_and(depth > min_depth, depth < max_depth)
    min_normalized_depth = depth[mask].min()
    max_normalized_depth = depth[mask].max()
    normalized_depth = (depth - min_normalized_depth) / (
        max_normalized_depth - min_normalized_depth
    )

    color = cm.jet(normalized_depth)
    u = u.astype(int)
    v = v.astype(int)

    image_filled = image0.copy()
    circle_radius = 8
    for i in range(len(u)):
        if (
            0 <= v[i] < image0.shape[0]
            and 0 <= u[i] < image0.shape[1]
            and min_depth < depth[i] < max_depth
        ):
            cv2.circle(
                image_filled, (u[i], v[i]), circle_radius, color[i, :3] * 255, -1
            )

    parent_dir = os.path.dirname(image_filled_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    image = Image.fromarray(image_filled)
    print(f"image_filled_path = {image_filled_path}\n")
    image.save(image_filled_path)


def process_all_pairs(root_path):
    sensor_aligment_path = os.path.join(
        root_path, "sensor_temporal_alignment_json", "000.json"
    )
    with open(sensor_aligment_path, "r") as f:
        sensor_aligment = json.load(f)

    for entry in sensor_aligment:
        for timestamp, data in entry.items():
            if "top_center_lidar" in data:
                pcd_path = os.path.join(root_path, data["top_center_lidar"])
                camera_path = os.path.join(root_path, data["center_camera_fov30"])

                proj_lidar(
                    root_path=root_path,
                    pcd_path=pcd_path,
                    camera_path=camera_path,
                    image_filled_key=os.path.basename(pcd_path).replace(".pcd", ""),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LiDAR and camera data.")
    parser.add_argument(
        "--root_path",
        type=str,
        required=True,
        help="Root path to the dataset.",
    )
    args = parser.parse_args()

    process_all_pairs(
        root_path=args.root_path,
    )