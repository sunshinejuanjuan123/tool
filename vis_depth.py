from os import read

import cv2
import numpy as np
from PIL import Image

# 读取 PNG 图像
# img = Image.open("/iag_ad_01/ad/yuanweizhong/datasets/shift_depth/0a03-8855/00000000_depth_front.png")
img = Image.open("/iag_ad_01/ad/yuanweizhong/datasets/shift_depth/0003-17fb/00000010_depth_front.png")
img = np.array(img).astype(np.float64)

# 解码为整数深度值
depth_raw = img[:, :, 0] + img[:, :, 1] * 256 + img[:, :, 2] * 256 * 256
import ipdb;ipdb.set_trace()
# 转换为米
depth_meters = depth_raw/16777216.0
depth_meters = depth_meters * 1000.0

# 可视化保存（仅用于显示）
depth_vis = cv2.normalize(depth_meters, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("decoded_depth_in_meters.png", depth_vis)

def visualize_sparse_depth(sparse_depth):
    # 设定无深度的像素为特定值（如-1），便于区分
    vis = sparse_depth.copy()
    vis[vis == 0] = -1
    # 归一化，仅对非零点
    valid_mask = vis > 0
    if np.any(valid_mask):
        minv, maxv = vis[valid_mask].min(), vis[valid_mask].max()
        norm = np.zeros_like(vis, dtype=np.float32)
        norm[valid_mask] = (vis[valid_mask] - minv) / (maxv - minv + 1e-8)
        norm = (norm * 255).astype(np.uint8)
        # 伪彩色
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        # 无深度点设为灰色
        color[vis == -1] = [128, 128, 128]
    else:
        color = np.full((*vis.shape, 3), 128, dtype=np.uint8)
    cv2.imwrite("sparse_depth_colormap.png", color)

K = np.array([
    [915,   0.0, 640.0],
    [  0.0, 1040, 400.0],
    [  0.0,   0.0,   1.0]
])

def save_sparse_depth_to_ply_with_intrinsics(sparse_depth, K, ply_path="sparse_depth.ply"):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = sparse_depth.shape
    points = []
    for v in range(h):
        for u in range(w):
            z = sparse_depth[v, u]
            if z > 0:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def read_low_depth(depth):
    depth= depth
    target_h, target_w = depth.shape[:2]
    # depth_lr = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    depth_lr = depth
    # 3. 构建扭曲网格采样 anchor 点
    stride = 16
    anchors = []
    for y in range(0, target_h, stride):
        for x in range(0, target_w, stride):
            dy = np.random.randint(-2, 3)
            dx = np.random.randint(-2, 3)
            yy = np.clip(y + dy, 0, target_h - 1)
            xx = np.clip(x + dx, 0, target_w - 1)
            anchors.append((yy, xx))
    anchors = np.array(anchors)

    # 只在 anchor 点保留真实深度，其余为0
    sparse_depth = np.zeros((target_h, target_w), dtype=depth_lr.dtype)
    sparse_depth[anchors[:,0], anchors[:,1]] = depth_lr[anchors[:,0], anchors[:,1]]

    # 可视化
    # visualize_sparse_depth(sparse_depth)
    # visualize_sparse_depth(depth_meters)
    breakpoint()
    save_sparse_depth_to_ply_with_intrinsics(sparse_depth, K)
    save_sparse_depth_to_ply_with_intrinsics(depth_meters, K, "original_depth.ply")

    return sparse_depth



read_low_depth(depth_meters)