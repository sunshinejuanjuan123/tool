import argparse
import json
import os
import re
from os.path import exists

import imageio
import numpy as np
import open3d as o3d
import torch
import tqdm
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor
from torchvision.utils import save_image

# 使用 argparse 读取 root_path
parser = argparse.ArgumentParser(description="Process point cloud and depth data.")
parser.add_argument('--root_path', type=str, required=True, help='Root path to the data directory.')
args = parser.parse_args()



root_path = args.root_path
sensor_aligment_path = root_path + "/sensor_temporal_alignment.json"
sensor_aligment = json.load(open(sensor_aligment_path))
intrinsic_path = root_path + '/calib/center_camera_fov30/center_camera_fov30-intrinsic.json'
lidar_dir = os.path.join(root_path, "lidar", "top_center_lidar")
ply_files = [os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith('.ply')]
output_dir = os.path.join(root_path, "lidar", "top_center_lidar_depth")
os.makedirs(output_dir,exist_ok=True)

for ply in ply_files:

    file_name = os.path.basename(ply)
    mmumeric_part=file_name[:19]
    
    pcd= o3d.io.read_point_cloud(ply)
    points = np.asarray(pcd.points)

    # 加载相机内参 json 文件
    intrinsic = json.load(open(intrinsic_path))
    # 获取相机内参矩阵
    intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])

    # 获取图像高度和宽度
    h, w = int(intrinsic['value0']['param']['img_dist_h']), int(intrinsic['value0']['param']['img_dist_w'])
    # 将内参矩阵赋值给 K
    K = intrinsic_np
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # 打印点云深度范围
    print("Depth range: ", points[:, 2].min(), points[:, 2].max())

    # 创建深度图像数组
    pts_depth = np.zeros([1, h, w])
    # 将点云数据赋值给 point_camera
    point_camera = points
    # 筛选出深度值大于 0 的点
    uvz = point_camera[point_camera[:, 2] > 0]
    # 将点云坐标转换到图像坐标系
    uvz = uvz @ K.T
    # 进行透视除法
    uvz[:, :2] /= uvz[:, 2:]
    # 筛选出在图像范围内的点
    uvz = uvz[uvz[:, 1] >= 0]
    uvz = uvz[uvz[:, 1] < h]
    uvz = uvz[uvz[:, 0] >= 0]
    uvz = uvz[uvz[:, 0] < w]
    # 获取图像坐标
    uv = uvz[:, :2]
    # 将坐标转换为整数
    uv = uv.astype(int)
    # 将深度值填充到深度图中
    pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]

    # 保存未插值的深度图
    pts_depth_uint16 = (pts_depth[0] * 1000)
    pts_depth_uint16[pts_depth_uint16 > 65535] = 65535
    pts_depth_uint16 = pts_depth_uint16.astype(np.uint16)
    Image.fromarray(pts_depth_uint16).save(os.path.join(output_dir, f'{mmumeric_part}.png'))
    np.savez(os.path.join(output_dir, f'{mmumeric_part}.npz'), depth=pts_depth[0])