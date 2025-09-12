import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

def interpolate_point_cloud(P1, t1, P2, t2, t_target, R_1_to_2, T_1_to_2):
    # # 假设已通过ICP或其他方法得到 R 和 T
    # R_1_to_2 = ...  # 旋转矩阵（3x3）
    # T_1_to_2 = ...  # 平移向量（3,）
    
    alpha = (t_target - t1) / (t2 - t1)
    
    # 旋转插值（Slerp）
    rot_1 = Rotation.from_matrix(np.eye(3))       # t1时刻的单位旋转
    rot_2 = Rotation.from_matrix(R_1_to_2)
    
    def slerp(rot_1, rot_2, alpha):
        # 将旋转对象转换为四元数
        q1 = rot_1.as_quat()
        q2 = rot_2.as_quat()
        
        # 计算四元数之间的点积
        dot = np.sum(q1 * q2)
        
        # 如果点积为负，取反q2以选择最短路径
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # 如果点积接近1，使用线性插值
        if dot > 0.9995:
            return Rotation.from_quat(q1 + alpha * (q2 - q1))
        
        # 计算插值角度
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        # 计算插值后的四元数
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        q = s0 * q1 + s1 * q2
        
        return Rotation.from_quat(q)

    rot_alpha = slerp(rot_1, rot_2, alpha)  # 插值后的旋转对象
    R_alpha = rot_alpha.as_matrix()
    
    # 平移插值
    T_alpha = alpha * T_1_to_2
    
    # 应用变换到P1
    Pt = (R_alpha @ P1.T).T + T_alpha
    return Pt

backend_pose_path = '/data/lidar_data_test/slam_output/map_generation_results/backend-pose.txt'
#ValueError: could not convert string '2024-09-08-15-53-23-299' to float64 at row 0, column 1.
backend_pose = np.loadtxt(backend_pose_path,dtype=str)
backend_pose_valid=backend_pose[:,1:]
#boxx.tree(backend_pose_valid[0])
# └── /: (12,)<U23
backend_pose_valid_float=backend_pose_valid.astype(np.float32)

R_1=backend_pose_valid_float[0].reshape(3,4)[:3,:3]
T_1=backend_pose_valid_float[0].reshape(3,4)[:3,3]

R_2=backend_pose_valid_float[1].reshape(3,4)[:3,:3]
T_2=backend_pose_valid_float[1].reshape(3,4)[:3,3]

R_1_to_2=R_2@np.linalg.inv(R_1)
T_1_to_2=T_2-R_1_to_2@T_1
t1=1725782003299000000
t2=1725782003399000000
t_target=1725782003299999850
# input_set = 'top_center_lidar_vehicle'
input_set = 'pvb_vehicle'
pcd_path1=f'/data/lidar_data_test/slam_output/map_generation_results/{input_set}/{t1}.pcd'
pcd1 = o3d.io.read_point_cloud(pcd_path1)
points1 = np.asarray(pcd1.points)

pcd_path2=f'/data/lidar_data_test/slam_output/map_generation_results/{input_set}/{t2}.pcd'
pcd2 = o3d.io.read_point_cloud(pcd_path2)
points2 = np.asarray(pcd2.points)


points_target=interpolate_point_cloud(points1, t1, points2, t2, t_target, R_1_to_2, T_1_to_2)
# import ipdb;ipdb.set_trace()
pcd_output=o3d.geometry.PointCloud()
pcd_output.points=o3d.utility.Vector3dVector(points_target)
o3d.io.write_point_cloud(pcd_path1.replace('.pcd', '_interpolated.ply').replace(str(t1),str(t_target)), pcd_output)













