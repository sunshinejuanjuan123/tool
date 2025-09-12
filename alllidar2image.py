import open3d as o3d
import numpy as np
import json
from matplotlib import cm
import cv2
import os
from PIL import Image
import re
from tqdm import tqdm
def find_closest_index(a_filename, folder_path):
  """
  在指定文件夹中查找与给定文件名最接近的 index 文件。

  Args:
    a_filename: 给定文件名，例如 "indexA.png"。
    folder_path: 文件夹路径。

  Returns:
    与给定文件名最接近的 index 文件名，例如 "indexB.png"，如果未找到则返回 None。
  """

  # 从文件名中提取数字部分
#   import ipdb;ipdb.set_trace()
  a_index = int(re.findall(r'\d+', a_filename)[-1])

  # 初始化最小差值和最接近的文件名
  min_diff = float('inf')
  closest_filename = None

  # 遍历文件夹中的所有文件
  for filename in os.listdir(folder_path):
    # 过滤掉非 index 文件
    # jpg 和 png 文件   
    if not filename.endswith('.jpeg') and not filename.endswith('.png') and not filename.endswith('.jpg'):
          print("not filename.endswith('.jpeg');not filename.endswith('.png')")
          continue

    # 从文件名中提取数字部分
    b_index = int(re.findall(r'\d+', filename)[-1])

    # 计算差值
    diff = abs(a_index - b_index)

    # 更新最小差值和最接近的文件名
    if diff < min_diff:
      min_diff = diff
      closest_filename = filename

  return closest_filename
#load json file
def project_oneimage(pcd_path, image0, intrinsic_path, extrinsic_path, image_filled_path, lidar_extrinsic_path, lidar2camera_fov30_extrinsic_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    ply_path=pcd_path.replace('.pcd', '.ply')
    # o3d.io.write_point_cloud(ply_path, pcd)
    
    intrinsic = json.load(open(intrinsic_path))
    extrinsic = json.load(open(extrinsic_path))
    intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])
    extrinsic_np = np.array(extrinsic['center_camera_fov30-to-car_center-extrinsic']['param']['sensor_calib']['data'])
    camera2car=extrinsic_np

    lidar_extrinsic = json.load(open(lidar_extrinsic_path))
    lidar_extrinsic_np = np.array(lidar_extrinsic['top_center_lidar-to-car_center-extrinsic']['param']['sensor_calib']['data'])
    lidar2car=lidar_extrinsic_np

    import ipdb;ipdb.set_trace()
    lidar2camera=lidar2car@np.linalg.inv(camera2car)

    K=intrinsic_np
    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points_camera_fov30 = (lidar2camera @ points_h.T).T
    points_camera_fov30 = points_camera_fov30[:, :3]/points_camera_fov30[:, 3:4]
    pcd_camera_fov30 = o3d.geometry.PointCloud()
    pcd_camera_fov30.points = o3d.utility.Vector3dVector(points_camera_fov30)
    o3d.io.write_point_cloud(ply_path.replace('.ply', '_camera_fov30.ply'), pcd_camera_fov30)
    u = K[0, 0] * points_camera_fov30[:, 0] / points_camera_fov30[:, 2] + K[0, 2]
    v = K[1, 1] * points_camera_fov30[:, 1] / points_camera_fov30[:, 2] + K[1, 2]
    depth = points_camera_fov30[:, 2]

    min_depth=0
    max_depth=1000
    mask = np.zeros_like(depth.shape[0])
    mask = np.logical_and(depth > min_depth, depth < max_depth)

# 归一化深度值
    min_normalized_depth = depth[mask].min()
    max_normalized_depth = depth[mask].max()
    normalized_depth = (depth - min_normalized_depth) / (max_normalized_depth - min_normalized_depth)


# 使用 jet 颜色映射
    color = cm.jet(normalized_depth)

    u = u.astype(int)
    v = v.astype(int)

    rgb_colors=[]
#将点云圆点画粗点

    for x,y,d in zip(u,v,depth):
        if 0 <= x < image0.shape[1] and 0 <= y < image0.shape[0] and min_depth<d<max_depth:
            rgb_colors.append(image0[y, x])
        else:
            rgb_colors.append([0, 0, 0])  # 如果投影点超出图像边界，则赋予黑色
    image_filled = image0.copy()
    circle_radius=8
# Check and fill the pixels at the (u, v) coordinates with red color
    for i in range(len(u)):
        if 0 <= v[i] < image0.shape[0] and 0 <= u[i] < image0.shape[1] and min_depth<depth[i]<max_depth:
        # image_filled[v[i], u[i]] = color[i,:3]*255
            cv2.circle(image_filled, (u[i],v[i]), circle_radius, color[i,:3]*255, -1) 


# image_filled_path = root_path + "/camera/center_camera_fov30/1725782003299999850_filled.jpg"


# 获取父文件夹路径
    parent_dir = os.path.dirname(image_filled_path)


# 检查父文件夹是否存在
    if not os.path.exists(parent_dir):
    # 如果不存在，则创建父文件夹
        os.makedirs(parent_dir)
# 打开图片
    image = Image.fromarray(image_filled)
# 保存图片
    image.save(image_filled_path)

root_path = '/data/2024_09_08_07_53_23_pathway_pilotGtParser'
# pcd_path = root_path + "/lidar/top_center_lidar/1725782003299999850.pcd"


# o3d.visualization.draw_geometries([pcd])
center_camera_fov30_path = root_path + "/camera/center_camera_fov30"
intrinsic_path = root_path + "/calib/center_camera_fov30/center_camera_fov30-intrinsic.json"
lidar2camera_fov30_extrinsic_path = root_path + "/calib/top_center_lidar/top_center_lidar-to-center_camera_fov30-extrinsic.json"

lidar_extrinsic_path = root_path + "/calib/top_center_lidar/top_center_lidar-to-car_center-extrinsic.json"
extrinsic_path = root_path + "/calib/center_camera_fov30/center_camera_fov30-to-car_center-extrinsic.json"

# pcd_path=lidars_path+'/1725782003299000000.pcd'
# pcd_path = root_path + "/lidar/top_center_lidar/1725782003299999850.pcd"
# lidars_path='/data/lidar_data_test/slam_output/map_generation_results/top_center_lidar_vehicle'
lidars_path=root_path+'/lidar/top_center_lidar/'

#遍历center_camera_fov30_path下的所有jpg文件
for lidar_path in tqdm(sorted(os.listdir(lidars_path)),desc='lidar'):
    if lidar_path.endswith('.pcd'):
        pcd_path=lidars_path+'/'+lidar_path
        camera_path=find_closest_index(lidar_path, center_camera_fov30_path)
        center_camera_fov30_image_path = center_camera_fov30_path + "/" + camera_path
        image0 = cv2.imread(center_camera_fov30_image_path)
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        # import ipdb;ipdb.set_trace()
        # break
        image_filled_path = root_path + "/camera/center_camera_fov30_filled/" + camera_path
        if not os.path.exists(image_filled_path):
            os.makedirs(os.path.dirname(image_filled_path), exist_ok=True)


        project_oneimage(pcd_path,  image0, intrinsic_path, extrinsic_path, image_filled_path, lidar_extrinsic_path, lidar2camera_fov30_extrinsic_path)






