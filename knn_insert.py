import imageio
import numpy as np
# from general_utils import load_ply
import torch
from torchvision.utils import save_image
from sklearn.neighbors import KNeighborsRegressor
# from promptda.utils.depth_utils import visualize_depth
from PIL import Image
import open3d as o3d
import json
import tqdm
# 定义点云文件路径
root_path = '/data/senseauto/高速远距离数据/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser/'
sensor_aligment_path = root_path + "/sensor_temporal_alignment.json/000.json"
sensor_aligment = json.load(open(sensor_aligment_path))
for k,v in sensor_aligment[1].items():
    ply_path = root_path + v['top_center_lidar'].replace('.pcd', '_camera_fov30.ply')
    # 定义相机内参文件路径
intrinsic_path = root_path + '/calib/center_camera_fov30/center_camera_fov30-intrinsic.json'

#  使用open3d读取点云文件
pcd = o3d.io.read_point_cloud(ply_path)
# 将点云数据转换为numpy数组
points = np.asarray(pcd.points)

# 加载相机内参json文件
intrinsic = json.load(open(intrinsic_path))
# 获取相机内参矩阵
intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])

# 获取图像高度和宽度
h, w = int(intrinsic['value0']['param']['img_dist_h']), int(intrinsic['value0']['param']['img_dist_w'])
# 将内参矩阵赋值给K
K = intrinsic_np
cx = K[0, 2]
cy = K[1, 2]
fx = K[0, 0]
fy = K[1, 1]
# h = int(cy*2)
# w = int(cx*2)
# 打印点云深度范围
print("Depth range: ", points[:, 2].min(), points[:, 2].max())

# 创建深度图像数组
pts_depth = np.zeros([1, h, w])
# 将点云数据赋值给point_camera
point_camera = points
# 筛选出深度值大于0的点
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

pts_depth_uint16 = (pts_depth[0]*1000)
pts_depth_uint16[pts_depth_uint16>65535]=65535
pts_depth_uint16 = pts_depth_uint16.astype(np.uint16)
Image.fromarray(pts_depth_uint16).save('pts_depth_origin.png')
# 将深度图转换为torch张量
pts_depth = torch.from_numpy(pts_depth).float()

# 使用KNN进行深度补全,K=4
# 找到所有有深度值的点
valid_points = []
valid_depths = []
# 遍历图像所有像素
for i in (range(h)):
    for j in (range(w)):
        # 如果该点有深度值
        if pts_depth[0, i, j] > 0:
            # 记录该点的坐标和深度值
            valid_points.append([i, j])
            valid_depths.append(pts_depth[0, i, j])

# 将列表转换为numpy数组
valid_points = np.array(valid_points)
valid_depths = np.array(valid_depths)

# 如果存在有效点
if len(valid_points) > 0:
    # 创建并训练KNN模型
    knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
    knn.fit(valid_points, valid_depths)
    
    # 找到所有没有深度值的点
    invalid_points = []
    for i in (range(h)):
        for j in (range(w)):
            if pts_depth[0, i, j] == 0:
                invalid_points.append([i, j])
    
    # 将列表转换为numpy数组
    invalid_points = np.array(invalid_points)
    
    # 如果存在无效点
    if len(invalid_points) > 0:
        # 使用KNN预测深度值
        pred_depths = knn.predict(invalid_points)
        
        # 将预测的深度值填充到深度图中
        for idx, (i, j) in (enumerate(invalid_points)):
            pts_depth[0, i, j] = pred_depths[idx]

pts_depth = pts_depth.numpy()
depth_int = (pts_depth[0] * 1000)
depth_int[depth_int>65535]=65535
depth_int = depth_int.astype(np.uint16)
print(depth_int.shape)
print(depth_int.min(), depth_int.max())
Image.fromarray(depth_int).save('pts_depth_knn.png')
print(pts_depth.shape)
print(pts_depth.min(), pts_depth.max())
# import ipdb;ipdb.set_trace()
np.savez('pts_depth.npz', depth=pts_depth[0])
# depth_vis, depth_min, depth_max = visualize_depth(pts_depth[0, :, :], ret_minmax=True)
# imageio.imwrite('pts_depth_vis.png', depth_vis)
