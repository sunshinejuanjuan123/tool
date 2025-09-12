import os
import argparse
from tqdm import tqdm
import json
import cv2
import numpy as np
import ast
from scipy.linalg import svd
import glob

def get_options():
    parser = argparse.ArgumentParser(description='Verify camera poses')
    parser.add_argument('--rgb_dir', type=str, required=True, help='Directory containing RGB images with timestamp names')
    parser.add_argument('--intrinsic_dir', type=str, required=True, help='Directory containing intrinsic files with timestamp names')
    parser.add_argument('--pose_dir', type=str, required=True, help='Directory containing pose files with timestamp names')
    parser.add_argument('--save_dir', type=str, default='verify_output', help='Output directory name')
    parser.add_argument('--mode2', action='store_true', help='Use detailed analysis mode')
    parser.add_argument('--image_scale', type=int, default=1, help='Image downscale factor')
    parser.add_argument('--angle_threshold', type=float, default=5.0, help='Angle threshold for frame selection')
    return parser.parse_args()

def extract_timestamp(filename):
    """从文件名中提取时间戳"""
    basename = os.path.splitext(filename)[0]
    try:
        return float(basename)
    except ValueError:
        # 如果不是纯数字，尝试其他格式
        import re
        timestamp_match = re.search(r'(\d+\.?\d*)', basename)
        if timestamp_match:
            return float(timestamp_match.group(1))
        return 0.0

def load_timestamped_data(rgb_dir, intrinsic_dir, pose_dir):
    """按时间戳加载所有数据"""
    # 获取所有文件
    rgb_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext.upper())))

    intrinsic_files = glob.glob(os.path.join(intrinsic_dir, '*.txt'))
    pose_files = glob.glob(os.path.join(pose_dir, '*.txt'))

    # 提取时间戳并排序
    rgb_data = [(extract_timestamp(os.path.basename(f)), f) for f in rgb_files]
    intrinsic_data = [(extract_timestamp(os.path.basename(f)), f) for f in intrinsic_files]
    pose_data = [(extract_timestamp(os.path.basename(f)), f) for f in pose_files]

    rgb_data.sort(key=lambda x: x[0])
    intrinsic_data.sort(key=lambda x: x[0])
    pose_data.sort(key=lambda x: x[0])

    print(f"Found {len(rgb_data)} RGB files, {len(intrinsic_data)} intrinsic files, {len(pose_data)} pose files")

    # 找到所有三种数据都存在的时间戳
    rgb_timestamps = set([x[0] for x in rgb_data])
    intrinsic_timestamps = set([x[0] for x in intrinsic_data])
    pose_timestamps = set([x[0] for x in pose_data])

    common_timestamps = rgb_timestamps & intrinsic_timestamps & pose_timestamps
    common_timestamps = sorted(list(common_timestamps))

    print(f"Found {len(common_timestamps)} common timestamps")

    if len(common_timestamps) < 2:
        raise ValueError("Need at least 2 frames with complete data (RGB + intrinsic + pose)")

    # 构建数据字典
    data_dict = {}
    for timestamp in common_timestamps:
        rgb_file = next(f for t, f in rgb_data if t == timestamp)
        intrinsic_file = next(f for t, f in intrinsic_data if t == timestamp)
        pose_file = next(f for t, f in pose_data if t == timestamp)

        data_dict[timestamp] = {
            'rgb': rgb_file,
            'intrinsic': intrinsic_file,
            'pose': pose_file
        }

    return data_dict, common_timestamps

def read_intrinsic_matrix(file_path):
    """读取3x3内参矩阵"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    for line in lines:
        if line.strip():
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)

    return np.array(matrix, dtype=np.float32)

def read_pose_matrix(file_path):
    """读取单个4x4姿态矩阵"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    for line in lines:
        if line.strip():
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)

    return np.array(matrix, dtype=np.float32)

def draw_epilines(img1_input, img2_input, lines, pts1, pts2,circle_size):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img1 = img1_input.copy()
    img2 = img2_input.copy()
    h, w, c = img1.shape
    
    try:
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            img1 = cv2.circle(img1, tuple(map(int, pt1)), circle_size, color, -1)
            if r[1] < 1e-6 and r[0] < 1e-6:
                print(f'{r} skip')
                continue
            elif r[1] < 1e-6:
                x0, y0 = map(int, [-r[2] / r[0], 0])
                x1, y1 = map(int, [-(r[2] + r[1] * h) / r[0], h])
            else:
                x0, y0 = map(int, [0, -r[2] / r[1]])
                x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
            # if x0 < 0 or x1 < 0 or y0 < 0 or y1 < 0:
            #     print(f'left top {x0} {x1}, {y0}, {y1} skip')
            #     continue
            # if x0 > w or x1 > w or y0 > h or y1 > h:
            #     print(f'right down {x0} {x1}, {y0}, {y1} skip')
            #     continue
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 2)

        full_image = np.concatenate((img1, img2), axis=1)
        return full_image
    except Exception as e:
        print(f"error: {e}. img1.shape: {img1.shape}, img2.shape: {img2.shape}")
        raise e

def compute_essential_matrix(R1, t1, R2, t2):
    # R = R2 @ R1.T  # 相对旋转
    # t = t2 - R @ t1  # 相对平移
    R = R1.T @ R2
    t = R1.T @ (t2 - t1)
    t_cross = np.array([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]])
    E = t_cross @ R
    return E

def compute_fundamental_matrix(E, K1, K2):
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

def fundamental_check(cv_image1, cv_image2, extrinsics1, extrinsics2, intrinsic1, intrinsic2, scene, folder):
    """基础矩阵验证函数"""
    fast_detector = cv2.FastFeatureDetector_create()
    corners1 = fast_detector.detect(cv_image1, None)

    if len(corners1) < 20:
        print(f"Not enough corners detected: {len(corners1)}")
        return

    R1 = extrinsics1[:3, :3]
    t1 = extrinsics1[:3, 3]
    R1 = R1.T
    t1 = -R1 @ t1

    R2 = extrinsics2[:3, :3]
    t2 = extrinsics2[:3, 3]
    R2 = R2.T
    t2 = -R2 @ t2

    K1 = intrinsic1.copy()
    K2 = intrinsic2.copy()

    print(f'K1:\n{K1}\nK2:\n{K2}\n{cv_image1.shape} -- {cv_image2.shape}')

    # 调整内参矩阵到图像尺寸
    if K1[0, 0] < 1.0:  # 归一化内参
        K1[0, 0] = intrinsic1[0, 0] * cv_image1.shape[1]
        K1[1, 1] = intrinsic1[1, 1] * cv_image1.shape[0]
        K1[0, 2] = intrinsic1[0, 2] * cv_image1.shape[1]
        K1[1, 2] = intrinsic1[1, 2] * cv_image1.shape[0]

    if K2[0, 0] < 1.0:  # 归一化内参
        K2[0, 0] = intrinsic2[0, 0] * cv_image2.shape[1]
        K2[1, 1] = intrinsic2[1, 1] * cv_image2.shape[0]
        K2[0, 2] = intrinsic2[0, 2] * cv_image2.shape[1]
        K2[1, 2] = intrinsic2[1, 2] * cv_image2.shape[0]

    print(f"R1: {R1}\nt1:{t1}\nR2: {R2}\nt2: {t2}\nK1: {K1}\nK2: {K2}")

    img_1_draw = cv_image1.copy()
    for corner in corners1:
        x, y = corner.pt
        cv2.circle(img_1_draw, (int(x), int(y)), 1, (0, 255, 0), -1)

    try:
        # 计算本质矩阵和基本矩阵
        E = compute_essential_matrix(R1, t1, R2, t2)
        F = compute_fundamental_matrix(E, K1, K2)

        print(f'E:\n{E}\nF:\n{F}')

        # 随机选择特征点
        num_corners = min(20, len(corners1))
        rand_index = np.random.choice(len(corners1), num_corners, replace=False)
        pts1 = []
        for i in rand_index:
            pt = corners1[i].pt
            pts1.append([pt[0], pt[1], 1])
        pts1 = np.array(pts1)

        # 计算极线
        lines = pts1 @ F.T  # 注意这里使用F.T
        circle_size = 40 * args.image_scale
        # 绘制极线
        img_epip_draw = draw_epilines(cv_image1, cv_image2, lines, pts1[:, :2], pts1[:, :2], circle_size)
        cv2.imwrite(f"{folder}/{scene}epilines_image1_2.png", img_epip_draw)
        cv2.imwrite(f"{folder}/{scene}corners_image_1.png", img_1_draw)
        print(f'Saved {folder}/{scene}corners_image_1.png')

    except Exception as e:
        print(f"Error in fundamental_check: {e}")
        print(f"Scene: {scene}, shapes - img1: {cv_image1.shape}, img2: {cv_image2.shape}")

def detailed_analysis_mode(data_dict, timestamps, args):
    """详细分析模式"""
    folder = os.path.join("./output", args.save_dir)
    os.makedirs(folder, exist_ok=True)

    # 选择第一帧
    timestamp1 = timestamps[0]
    data1 = data_dict[timestamp1]

    # 加载第一帧数据
    image1 = cv2.imread(data1['rgb'])
    if args.image_scale > 1:
        h, w = image1.shape[:2]
        image1 = cv2.resize(image1, (w // args.image_scale, h // args.image_scale), interpolation=cv2.INTER_AREA)

    intrinsic1 = read_intrinsic_matrix(data1['intrinsic'])
    pose1 = read_pose_matrix(data1['pose'])
    print(f'pose1:{pose1}')

    h, w, c = image1.shape
    Rwc1 = pose1[:3, :3].T

    # 寻找合适的第二帧
    for i in range(1, len(timestamps)):
        timestamp2 = timestamps[i]
        data2 = data_dict[timestamp2]

        # 加载第二帧数据
        image2 = cv2.imread(data2['rgb'])
        if args.image_scale > 1:
            image2 = cv2.resize(image2, (w, h), interpolation=cv2.INTER_AREA)

        pose2 = read_pose_matrix(data2['pose'])
        Rwc2 = pose2[:3, :3].T

        # 计算角度差
        dR_12 = Rwc1 @ Rwc2.T
        d_angle = np.arccos(np.clip((np.trace(dR_12) - 1) / 2, -1, 1))
        d_angle_degree = d_angle * 180 / np.pi

        print(f'Timestamp {timestamp1:.6f} - {timestamp2:.6f} angle: {d_angle_degree:.2f} degrees')

        if d_angle_degree > args.angle_threshold and d_angle_degree < 180 - args.angle_threshold:
            break
    else:
        print("No suitable second frame found")
        return

    intrinsic2 = read_intrinsic_matrix(data2['intrinsic'])

    # 生成输出文件名
    output_prefix = f'{timestamp1:.6f}-{timestamp2:.6f}-'

    # 执行基础矩阵检查
    fundamental_check(image1, image2, pose1, pose2, intrinsic1, intrinsic2, output_prefix, folder)

    # ORB特征匹配和三角化
    if image1.shape[0] > 0 and image1.shape[1] > 0 and image2.shape[0] > 0 and image2.shape[1] > 0:
        orb_detector = cv2.ORB_create()
        kp1, des1 = orb_detector.detectAndCompute(image1, None)
        kp2, des2 = orb_detector.detectAndCompute(image2, None)

        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches12 = bf.match(des1, des2)
            print(f'Found {len(matches12)} initial matches')

            if len(matches12) > 8:
                # 基础矩阵过滤外点
                pts1 = np.array([kp1[match.queryIdx].pt for match in matches12])
                pts2 = np.array([kp2[match.trainIdx].pt for match in matches12])

                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

                # 保留内点
                good_matches = []
                pts1_filtered = []
                pts2_filtered = []
                kpts1_filtered = []
                kpts2_filtered = []

                for i, match in enumerate(matches12):
                    if mask[i] == 1:
                        pts1_filtered.append(pts1[i])
                        pts2_filtered.append(pts2[i])
                        kpts1_filtered.append(kp1[match.queryIdx])
                        kpts2_filtered.append(kp2[match.trainIdx])
                        good_matches.append(cv2.DMatch(len(kpts1_filtered)-1, len(kpts2_filtered)-1, 0))

                print(f'After filtering: {len(good_matches)} good matches')

                # 绘制匹配结果
                img_matches = cv2.drawMatches(image1, kpts1_filtered, image2, kpts2_filtered, 
                                            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(f'{folder}/{output_prefix}matches.png', img_matches)

                if len(pts1_filtered) > 4:
                    # 调整内参矩阵
                    K1_scaled = intrinsic1.copy()
                    K2_scaled = intrinsic2.copy()

                    if K1_scaled[0, 0] < 1.0:  # 归一化内参
                        K1_scaled[0, 0] *= w
                        K1_scaled[1, 1] *= h
                        K1_scaled[0, 2] *= w
                        K1_scaled[1, 2] *= h

                    if K2_scaled[0, 0] < 1.0:  # 归一化内参
                        K2_scaled[0, 0] *= w
                        K2_scaled[1, 1] *= h
                        K2_scaled[0, 2] *= w
                        K2_scaled[1, 2] *= h

                    # 三角化
                    proj_mat1 = K1_scaled @ pose1[:3, :]
                    proj_mat2 = K2_scaled @ pose2[:3, :]

                    pts1_array = np.array(pts1_filtered).T
                    pts2_array = np.array(pts2_filtered).T

                    pts4d = cv2.triangulatePoints(proj_mat1, proj_mat2, pts1_array, pts2_array)
                    pts4d /= pts4d[3, :]
                    pts3d = pts4d[:3, :].T

                    # 3D重投影验证
                    img_3d_draw = np.concatenate((image1, image2), axis=1)
                    depth1 = []
                    depth2 = []

                    for pt_index in range(pts3d.shape[0]):
                        pt = pts3d[pt_index, :]
                        Xc_1 = pose1[:3, :3] @ pt + pose1[:3, 3]
                        Xc_2 = pose2[:3, :3] @ pt + pose2[:3, 3]

                        if Xc_1[2] > 0 and Xc_2[2] > 0:  # 确保在相机前方
                            depth1.append(Xc_1[2])
                            depth2.append(Xc_2[2])

                            pt_proj_1 = K1_scaled @ Xc_1
                            pt_proj_2 = K2_scaled @ Xc_2
                            pt_proj_1 /= pt_proj_1[2]
                            pt_proj_2 /= pt_proj_2[2]

                            # 绘制重投影点和原始特征点
                            # 在第一帧图像上绘制三维点重投影的圆点，绿色，半径为2
                            circle_size = 40 * args.image_scale
                            cv2.circle(img_3d_draw, (int(pt_proj_1[0]), int(pt_proj_1[1])), circle_size, (0, 255, 0), -1)
                            # 获取第一帧中对应的原始特征点坐标
                            pt_1 = pts1_filtered[pt_index]
                            # 在第一帧图像上绘制原始特征点，青色，半径为2
                            cv2.circle(img_3d_draw, (int(pt_1[0]), int(pt_1[1])), circle_size, (255, 255, 0), -1)
                            # 在第一帧图像上连接重投影点和原始特征点，红色线
                            cv2.line(img_3d_draw, (int(pt_proj_1[0]), int(pt_proj_1[1])), 
                                    (int(pt_1[0]), int(pt_1[1])), (0, 0, 255), 1)

                            # 计算第二帧重投影点在拼接图像中的坐标（加上宽度偏移）
                            pt_proj_2_draw = pt_proj_2 + np.array([w, 0, 0])
                            # 计算第二帧原始特征点在拼接图像中的坐标（加上宽度偏移）
                            pt_2_draw = pts2_filtered[pt_index] + np.array([w, 0])
                            # 在第二帧图像上绘制三维点重投影的圆点，绿色，半径为2
                            cv2.circle(img_3d_draw, (int(pt_proj_2_draw[0]), int(pt_proj_2_draw[1])), circle_size, (0, 255, 0), -1)
                            # 在第二帧图像上绘制原始特征点，青色，半径为2
                            cv2.circle(img_3d_draw, (int(pt_2_draw[0]), int(pt_2_draw[1])), circle_size, (255, 255, 0), -1)
                            # 在第二帧图像上连接重投影点和原始特征点，红色线
                            cv2.line(img_3d_draw, (int(pt_proj_2_draw[0]), int(pt_proj_2_draw[1])), 
                                    (int(pt_2_draw[0]), int(pt_2_draw[1])), (0, 0, 255), 1)

                            # 在拼接图像上连接第一帧和第二帧的原始特征点，黄色线
                            cv2.line(img_3d_draw, (int(pt_1[0]), int(pt_1[1])), 
                                    (int(pt_2_draw[0]), int(pt_2_draw[1])), (0, 255, 255), 1)

                    cv2.imwrite(f'{folder}/{output_prefix}3d.png', img_3d_draw)
                    print(f'Saved {folder}/{output_prefix}3d.png')

                    # 深度统计
                    if len(depth1) > 0:
                        depth1 = np.array(depth1)
                        depth2 = np.array(depth2)

                        print(f'Reconstructed {len(depth1)} 3D points')
                        print(f'Depth range - Frame1: {depth1.min():.2f} to {depth1.max():.2f}')
                        print(f'Depth range - Frame2: {depth2.min():.2f} to {depth2.max():.2f}')

def batch_verify_mode(data_dict, timestamps, args):
    """批量验证模式"""
    folder = os.path.join("./output", args.save_dir)
    os.makedirs(folder, exist_ok=True)

    frame_interval = max(1, len(timestamps) // 10)  # 间隔采样
    sample_count = min(5, len(timestamps) // 2)

    for sample in range(sample_count):
        idx1 = sample * frame_interval
        idx2 = min(idx1 + frame_interval, len(timestamps) - 1)

        if idx1 >= len(timestamps) or idx2 >= len(timestamps):
            break

        timestamp1 = timestamps[idx1]
        timestamp2 = timestamps[idx2]

        data1 = data_dict[timestamp1]
        data2 = data_dict[timestamp2]

        # 加载数据
        image1 = cv2.imread(data1['rgb'])
        image2 = cv2.imread(data2['rgb'])

        if args.image_scale > 1:
            h, w = image1.shape[:2]
            image1 = cv2.resize(image1, (w // args.image_scale, h // args.image_scale), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (w // args.image_scale, h // args.image_scale), interpolation=cv2.INTER_AREA)

        intrinsic1 = read_intrinsic_matrix(data1['intrinsic'])
        intrinsic2 = read_intrinsic_matrix(data2['intrinsic'])
        pose1 = read_pose_matrix(data1['pose'])
        pose2 = read_pose_matrix(data2['pose'])

        output_prefix = f'{timestamp1:.6f}-{timestamp2:.6f}-batch-'

        fundamental_check(image1, image2, pose1, pose2, intrinsic1, intrinsic2, output_prefix, folder)

if __name__ == "__main__":
    args = get_options()

    # 加载按时间戳命名的数据
    data_dict, timestamps = load_timestamped_data(args.rgb_dir, args.intrinsic_dir, args.pose_dir)

    if args.mode2:
        detailed_analysis_mode(data_dict, timestamps, args)
    else:
        batch_verify_mode(data_dict, timestamps, args)