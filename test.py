import numpy as np
import cv2
from PIL import Image
import os

def depth_rgb_to_ply(depth_path, rgb_path, output_path, fx, fy, cx, cy, depth_scale=1):
    """
    将深度图和RGB图像转换为PLY点云文件
    
    参数:
    depth_path: npy深度文件路径
    rgb_path: RGB图像路径
    output_path: 输出PLY文件路径
    fx, fy: 相机焦距参数
    cx, cy: 相机主点参数
    depth_scale: 深度缩放因子 (通常为1000.0，将毫米转换为米)
    """
    
    # 读取深度图
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # 深度值已经是厘米单位，直接转换为float32
    depth = depth.astype(np.float32)
    depth = depth / 100.0
    
    # 读取RGB图像
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # 确保深度图和RGB图像尺寸一致
    if depth.shape[:2] != rgb_img.shape[:2]:
        rgb_img = cv2.resize(rgb_img, (depth.shape[1], depth.shape[0]))
    
    height, width = depth.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 过滤有效深度值（大于0）
    valid_mask = depth > 0
    
    # 获取有效像素的坐标和深度值
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask] / depth_scale  # 转换为米
    
    # 将像素坐标转换为3D坐标
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    # 获取对应的RGB颜色
    r = rgb_img[valid_mask][:, 0]
    g = rgb_img[valid_mask][:, 1]
    b = rgb_img[valid_mask][:, 2]
    
    # 创建点云数组
    points = np.column_stack((x, y, z, r, g, b))
    
    # 写入PLY文件
    write_ply(output_path, points)
    
    print(f"点云已保存到: {output_path}")
    print(f"点云包含 {len(points)} 个点")

def write_ply(filename, points):
    """
    写入PLY格式点云文件
    
    参数:
    filename: 输出文件名
    points: 点云数据 (N, 6) - [x, y, z, r, g, b]
    """
    
    with open(filename, 'w') as f:
        # 写入PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 写入点云数据
        for point in points:
            x, y, z, r, g, b = point
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def batch_convert(depth_dir, rgb_dir, output_dir, fx, fy, cx, cy, depth_scale=1000.0):
    """
    批量转换深度图和RGB图像为点云
    
    参数:
    depth_dir: 深度文件目录
    rgb_dir: RGB图像目录
    output_dir: 输出目录
    fx, fy, cx, cy: 相机参数
    depth_scale: 深度缩放因子
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有npy文件
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
    
    for depth_file in depth_files:
        # 构建文件路径
        depth_path = os.path.join(depth_dir, depth_file)
        
        # 假设RGB文件名与深度文件名对应（可能需要调整）
        rgb_name = depth_file.replace('.npy', '.jpg')  # 或 .png
        if not os.path.exists(os.path.join(rgb_dir, rgb_name)):
            rgb_name = depth_file.replace('.npy', '.png')
        
        rgb_path = os.path.join(rgb_dir, rgb_name)
        
        if not os.path.exists(rgb_path):
            print(f"找不到对应的RGB图像: {rgb_path}")
            continue
        
        # 输出PLY文件路径
        ply_name = depth_file.replace('.npy', '.ply')
        output_path = os.path.join(output_dir, ply_name)
        
        try:
            depth_rgb_to_ply(depth_path, rgb_path, output_path, fx, fy, cx, cy, depth_scale)
        except Exception as e:
            print(f"处理文件 {depth_file} 时出错: {e}")

# 使用示例
if __name__ == "__main__":
    # 相机内参 - 根据提供的相机参数更新
    fx = 1.150790771484375000e+03   # 焦距x (cam_K[0][0])
    fy = 1.150790771484375000e+03   # 焦距y (cam_K[1][1])
    cx = 3.840000000000000000e+02  # 主点x坐标 (cam_K[0][2])
    cy = 2.160000000000000000e+02 # 主点y坐标 (cam_K[1][2])
    depth_scale = 1000.0    # 深度缩放因子 (需要根据你的深度图单位调整)
    
    # 单个文件转换示例
    depth_file = "/iag_ad_01/ad/yuanweizhong/datasets/vkitti/Scene02/morning/frames/depth/Camera_0/depth_00000.png"  # 你的深度文件路径
    # rgb_file = "/iag_ad_01/ad/yuanweizhong/vggt/examples/driver/images/1725782003299999850.jpg"      # 对应的RGB图像路径
    rgb_file = "/iag_ad_01/ad/yuanweizhong/datasets/vkitti/Scene02/morning/frames/rgb/Camera_0/rgb_00000.jpg"      # 对应的RGB图像路径
    output_file = "pointcloud.ply"  # 输出点云文件路径
    
    # 检查文件是否存在
    if os.path.exists(depth_file) and os.path.exists(rgb_file):
        depth_rgb_to_ply(depth_file, rgb_file, output_file, fx, fy, cx, cy, depth_scale)
    else:
        print("请确保深度文件和RGB文件存在，并更新文件路径")
        
        # 批量转换示例
        # depth_directory = "depth_images"    # 深度图目录
        # rgb_directory = "rgb_images"        # RGB图像目录  
        # output_directory = "pointclouds"    # 输出点云目录
        # batch_convert(depth_directory, rgb_directory, output_directory, fx, fy, cx, cy, depth_scale)