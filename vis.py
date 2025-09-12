import cv2
import numpy as np
import open3d as o3d
import os
import re

# from ITo3D_inf.scene import Scene
# from ITo3D_inf.ito3d_utils.graphics_utils import focal2fov, fov2focal
# from ITo3D_inf.scene.cameras import Camera
# from utils import array_to_video
import threading  
import shutil

def replace_start_end_frames_by_warp_imgs(num_circle, output_dir, num_images=46):
    replace_out_dir = os.path.join(output_dir, "rgb_replace")
    if not os.path.exists(replace_out_dir):
        os.makedirs(replace_out_dir)
    rgb_dir = os.path.join(output_dir, "rgb")
    warp_dir = os.path.join(output_dir, "rgb_warp")
    for i in range(num_circle):
        for j in range(num_images):
            warp_img_0 = f"{warp_dir}/{i * num_images}.png"
            warp_img_45 = f"{warp_dir}/{i * num_images + num_images - 1}.png"
            if j == 0:
                shutil.copy(warp_img_0, os.path.join(replace_out_dir, f"{i * num_images}.png"))
            elif j == num_images - 1:
                shutil.copy(warp_img_45, os.path.join(replace_out_dir, f"{i * num_images + num_images - 1}.png"))
            else:
                rgb_img_j = f"{rgb_dir}/{i * num_images + j}.png"
                shutil.copy(rgb_img_j, os.path.join(replace_out_dir, f"{i * num_images + j}.png"))

def load_images(rgb_path, depth_path, data_type="png", scale=0.1):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"无法读取图像: {rgb_path}")
    
    if data_type == "png":
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  
        depth = depth.astype(np.float32) / 100.0  # 将深度值转换为米
    elif data_type == "exr":
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图像: {depth_path}")

    if depth.shape[0] != rgb.shape[0] or depth.shape[1] != rgb.shape[1]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

    if len(depth.shape) > 2:
        depth = depth[:, :, 0]

    print(f"depth shape: {depth.shape}, rgb shape: {rgb.shape}")

    return rgb, depth



def calculate_radius_from_depth(depth, filter_max_depth=100):
    sample_depths = []
    for i in range(depth.shape[0]): 
        for j in range(depth.shape[1]): 
            if depth[i, j] > 0 and depth[i, j] < filter_max_depth:
                sample_depths.append(depth[i, j])
    average_depth = np.mean(sample_depths)
    radius = average_depth * 0.3
    return radius


def create_curved_trajectory_with_poses_forward_backward_circle(depth, filter_max_depth=100):
    radius = calculate_radius_from_depth(depth, filter_max_depth=filter_max_depth)
    print(f"radius: {radius}")

    delta_angle = 15
    arc_height = radius * 0.05
    
    poses = []
    num_intermediate_points = 20
    num_rotations = int(360 / delta_angle) 
    
    for rotation_idx in range(num_rotations):
        base_angle = rotation_idx * delta_angle * (np.pi / 180)

        y_axis = (0, 0, -1)
        z_axis = (np.cos(base_angle), np.sin(base_angle), 0)
        x_axis = np.cross(y_axis, z_axis)
        forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        # Forward movement along arc
        for j in range(num_intermediate_points + 1):
            t = j / num_intermediate_points
        
            # 计算水平弧线上的点（偏移在水平面内）
            forward_dist = radius * t
            arc_offset = arc_height * np.sin(t * np.pi)  # 水平弧线偏移
            # 计算最终位置：基础直线位置 + 垂直于移动方向的偏移
            perp_angle = base_angle + np.pi/2  # 垂直于基础移动方向的角度
            pos = np.array([
                forward_dist * np.cos(base_angle) + arc_offset * np.cos(perp_angle),
                forward_dist * np.sin(base_angle) + arc_offset * np.sin(perp_angle),
                0  # 保持在水平面内
            ])

            # 计算切线方向
            tangent = np.array([
                -radius * np.sin(base_angle),  # 切线方向的x分量
                radius * np.cos(base_angle),    # 切线方向的y分量
                0
            ])
            z_axis = tangent / np.linalg.norm(tangent)  # 归一化切线方向
            
            # 计算下一个点的位置
            if j < num_intermediate_points:  # 确保不超出范围
                next_t = (j + 1) / num_intermediate_points
                next_forward_dist = radius * next_t
                next_arc_offset = arc_height * np.sin(next_t * np.pi)
                next_pos = np.array([
                    next_forward_dist * np.cos(base_angle) + next_arc_offset * np.cos(perp_angle),
                    next_forward_dist * np.sin(base_angle) + next_arc_offset * np.sin(perp_angle),
                    0
                ])
                # 计算下一个点的方向
                direction = next_pos - pos
                z_axis = direction / np.linalg.norm(direction)  # 使用下一个点的方向更新z轴
            
            y_axis = (0, 0, -1)  # 保持y轴向下
            x_axis = np.cross(y_axis, z_axis)  # 计算x轴
            x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化x轴
            y_axis = np.cross(z_axis, x_axis)  # 重新计算y轴确保正交性
            
            forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

            poses.append((forward_rotation, pos))
        
        # Backward movement along arc
        for j in range(num_intermediate_points-1, -1, -1):
            t = j / num_intermediate_points
            # 计算水平弧线上的点，注意这里反向偏移（使用 -arc_offset）
            forward_dist = radius * t
            arc_offset = -arc_height * np.sin(t * np.pi)  # 反向偏移
            perp_angle = base_angle + np.pi/2
            pos = np.array([
                forward_dist * np.cos(base_angle) + arc_offset * np.cos(perp_angle),
                forward_dist * np.sin(base_angle) + arc_offset * np.sin(perp_angle),
                0
            ])

            # 计算切线方向
            tangent = np.array([
                -radius * np.sin(base_angle),  # 切线方向的x分量
                radius * np.cos(base_angle),    # 切线方向的y分量
                0
            ])
            z_axis = -tangent / np.linalg.norm(tangent)  # 反向切线方向
            
            # 计算下一个点的位置
            if j > 0:  # 确保不超出范围
                next_t = (j - 1) / num_intermediate_points
                next_forward_dist = radius * next_t
                next_arc_offset = -arc_height * np.sin(next_t * np.pi)
                next_pos = np.array([
                    next_forward_dist * np.cos(base_angle) + next_arc_offset * np.cos(perp_angle),
                    next_forward_dist * np.sin(base_angle) + next_arc_offset * np.sin(perp_angle),
                    0
                ])
                # 计算下一个点的方向
                direction = next_pos - pos
                z_axis = direction / np.linalg.norm(direction)  # 使用下一个点的方向更新z轴
            
            y_axis = (0, 0, -1)  # 保持y轴向下
            x_axis = np.cross(y_axis, z_axis)  # 计算x轴
            x_axis = x_axis / np.linalg.norm(x_axis)  # 归一化x轴
            y_axis = np.cross(z_axis, x_axis)  # 重新计算y轴确保正交性
            
            backward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
            poses.append((backward_rotation, pos))

        # Smooth rotation to the next base angle
        if rotation_idx < num_rotations - 1:
            next_base_angle = (rotation_idx + 1) * delta_angle * (np.pi / 180)
            next_z_axis = (np.cos(next_base_angle), np.sin(next_base_angle), 0)
            next_x_axis = np.cross(y_axis, next_z_axis)
            next_forward_rotation = np.stack([next_x_axis, y_axis, next_z_axis], axis=1)

            # Calculate the angle difference
            angle_diff = np.degrees(np.arccos(np.clip(np.dot(z_axis, next_z_axis), -1.0, 1.0)))
            
            # Determine the number of interpolation steps based on angle difference
            max_angle_diff = 2  # degrees
            num_interpolations = max(8, int(np.ceil(angle_diff / max_angle_diff)))  # Ensure at least 8 interpolations

            # Interpolate between current and next rotation
            for interp_idx in range(num_interpolations):
                interp_t = interp_idx / num_interpolations
                interpolated_rotation = (1 - interp_t) * forward_rotation + interp_t * next_forward_rotation
                poses.append((interpolated_rotation, pos))

    return poses

def create_curved_trajectory_with_poses_forward_backward_rotation(depth, delta=60):
    poses = []
    num_intermediate_points = 22
    delta_z = delta / num_intermediate_points

    # 2. 正转
    start_angle = 90

    # 正转
    for j in range(num_intermediate_points + 1):
        cur_angle = (start_angle - j * delta_z) * (np.pi / 180)
        y_axis = (0, 0, -1)
        z_axis = (np.cos(cur_angle), np.sin(cur_angle), 0)
        x_axis = np.cross(y_axis, z_axis)
        forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        poses.append((forward_rotation, (0, 0, 0)))

    # 反转
    for j in range(num_intermediate_points, -1, -1):
        cur_angle = (start_angle - j * delta_z) * (np.pi / 180)
        y_axis = (0, 0, -1)
        z_axis = (np.cos(cur_angle), np.sin(cur_angle), 0)
        x_axis = np.cross(y_axis, z_axis)
        backward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        poses.append((backward_rotation, (0, 0, 0)))

    return poses

def create_curved_trajectory_with_poses_forward_backward_circle_wide(depth, delta_fov = 120, num_circle=3, two_layer_flag=False, use_radius_list=False):
    base_radius = depth[depth.shape[0]//2, depth.shape[1]//2]
    radius_list = [base_radius * 0.7, 
                   base_radius * 0.6,
                   base_radius * 1.2,
                   base_radius * 1.0,
                   ]
    print(f"radius: {base_radius}")
    delta_angle = delta_fov/num_circle
    print(f"delta_angle: {delta_angle}, num_circle: {num_circle}, delta_fov: {delta_fov}")
    
    poses = []
    num_intermediate_points = 22
    num_rotations = int(360 / delta_angle) 
    delta_z = delta_angle/num_intermediate_points/2
    print(f"delta_z: {delta_z}")
    # 2. 前后移动
    start_angle = 90 + delta_fov/2
    end_angle = start_angle - (num_circle-1) * delta_angle - 1
    print(f"start_angle: {start_angle}, end_angle: {end_angle}, delta_angle: {delta_angle}, delta_fov: {delta_fov}")
    layer_start_pos_list = []
    layer_start_angle_list = []
    for rotation_idx in range(num_rotations):
        # forward movement
        base_angle_degree = start_angle - rotation_idx * delta_angle
        base_angle = base_angle_degree * (np.pi / 180)

        if base_angle_degree < end_angle:
            continue

        if use_radius_list:
            radius = radius_list[rotation_idx]
            arc_height = radius * 0.05
        else:
            radius = base_radius * 0.3
            arc_height = radius * 0.05

        print(f"base_angle_degree: {base_angle_degree}, end_angle: {end_angle}")
        print(f"radius: {radius}, rotation_idx: {rotation_idx}")

        for j in range(num_intermediate_points +1):
            t = j / num_intermediate_points
            cur_angle = (base_angle_degree - j * delta_z) * (np.pi / 180)
            y_axis = (0, 0, -1)
            z_axis = (np.cos(cur_angle), np.sin(cur_angle), 0)
            x_axis = np.cross(y_axis, z_axis)
            forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

            # 计算水平弧线上的点（偏移在水平面内）
            forward_dist = radius * t
            arc_offset = arc_height * np.sin(t * np.pi)  # 水平弧线偏移
            # 计算最终位置：基础直线位置 + 垂直于移动方向的偏移
            perp_angle = base_angle + np.pi/2  # 垂直于基础移动方向的角度
            pos = np.array([
                forward_dist * np.cos(base_angle) + arc_offset * np.cos(perp_angle),
                forward_dist * np.sin(base_angle) + arc_offset * np.sin(perp_angle),
                0  # 保持在水平面内
            ])
            poses.append((forward_rotation, pos))
            #print(f"forward cur_angle: {base_angle_degree - j * delta_z}, j: {j}")

        layer_start_pos_list.append(pos)
        layer_start_angle_list.append(base_angle_degree)
        #print(f"forward poses: {len(poses)}")
        
        # Backward movement along arc
        for j in range(num_intermediate_points, -1, -1):
            t = j / num_intermediate_points

            cur_angle = (base_angle_degree - (num_intermediate_points *2 -j) * delta_z) * (np.pi / 180)
            y_axis = (0, 0, -1)
            z_axis = (np.cos(cur_angle), np.sin(cur_angle), 0)
            x_axis = np.cross(y_axis, z_axis)
            backward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

            # 计算水平弧线上的点，注意这里反向偏移（使用 -arc_offset）
            forward_dist = radius * t
            arc_offset = -arc_height * np.sin(t * np.pi)  # 反向偏移
            perp_angle = base_angle + np.pi/2
            pos = np.array([
                forward_dist * np.cos(base_angle) + arc_offset * np.cos(perp_angle),
                forward_dist * np.sin(base_angle) + arc_offset * np.sin(perp_angle),
                0
            ])
            poses.append((backward_rotation, pos))
            #print(f"backward cur_angle: {base_angle_degree - (num_intermediate_points *2 -j) * delta_z}, j: {j}")
        #print(f"backward poses: {len(poses)}")

    if not two_layer_flag:
        return poses

    # layer 2
    num_circle = 3
    for k in range(len(layer_start_pos_list)):
        layer_start_pos = layer_start_pos_list[k]
        layer_start_angle_degree = layer_start_angle_list[k] + delta_angle

        for i in range(num_circle):
            base_angle_degree =  layer_start_angle_degree - i * delta_angle
            base_angle = base_angle_degree * (np.pi / 180)

            for j in range(num_intermediate_points +1):
                t = j / num_intermediate_points
                cur_angle = (base_angle_degree - j * delta_z) * (np.pi / 180)
                y_axis = (0, 0, -1)
                z_axis = (np.cos(cur_angle), np.sin(cur_angle), 0)
                x_axis = np.cross(y_axis, z_axis)
                forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

                # 计算水平弧线上的点（偏移在水平面内）
                forward_dist = radius * t
                arc_offset = arc_height * np.sin(t * np.pi)  # 水平弧线偏移
                # 计算最终位置：基础直线位置 + 垂直于移动方向的偏移
                perp_angle = base_angle + np.pi/2  # 垂直于基础移动方向的角度
                pos = np.array([
                    forward_dist * np.cos(base_angle) + arc_offset * np.cos(perp_angle),
                    forward_dist * np.sin(base_angle) + arc_offset * np.sin(perp_angle),
                    0  # 保持在水平面内
                ])
                print(f"layer 2 forward cur_angle: {base_angle_degree - j * delta_z}, j: {j}")
                poses.append((forward_rotation, pos + layer_start_pos))
            print(f"layer 2 forward poses: {len(poses)}")
        
            # Backward movement along arc
            for j in range(num_intermediate_points, -1, -1):
                t = j / num_intermediate_points

                cur_angle = (base_angle_degree - (num_intermediate_points *2 -j) * delta_z) * (np.pi / 180)
                y_axis = (0, 0, -1)
                z_axis = (np.cos(cur_angle), np.sin(cur_angle), 0)
                x_axis = np.cross(y_axis, z_axis)
                backward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

                # 计算水平弧线上的点，注意这里反向偏移（使用 -arc_offset）
                forward_dist = radius * t
                arc_offset = -arc_height * np.sin(t * np.pi)  # 反向偏移
                perp_angle = base_angle + np.pi/2
                pos = np.array([
                    forward_dist * np.cos(base_angle) + arc_offset * np.cos(perp_angle),
                    forward_dist * np.sin(base_angle) + arc_offset * np.sin(perp_angle),
                    0
                ])
                poses.append((backward_rotation, pos + layer_start_pos))
                print(f"layer 2 backward cur_angle: {base_angle_degree - (num_intermediate_points *2 -j) * delta_z}, j: {j}")
            print(f"layer 2 backward poses: {len(poses)}")

    return poses
 

def create_curved_trajectory_with_poses_forward_backward(depth, filter_max_depth=100):
    radius = calculate_radius_from_depth(depth, filter_max_depth=filter_max_depth)
    print(f"radius: {radius}")

    delta_angle = 15
    
    poses = []
    num_intermediate_points = 5  # Points for smooth transition
    num_rotations = int(360 / delta_angle)  # Number of different angles (360/45 = 8)
    
    for rotation_idx in range(num_rotations):
        base_angle = rotation_idx * delta_angle * (np.pi / 180)
        
        y_axis = (0, 0, -1)
        z_axis = (np.cos(base_angle), np.sin(base_angle), 0)
        x_axis = np.cross(y_axis, z_axis)
        forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        # Forward movement
        for j in range(num_intermediate_points + 1):
            t = j / num_intermediate_points
            pos = np.array([
                radius * t * np.cos(base_angle),
                radius * t * np.sin(base_angle),
                0
            ])
            poses.append((forward_rotation, pos))
        
        # Backward movement
        for j in range(num_intermediate_points, -1, -1):
            t = j / num_intermediate_points
            pos = np.array([
                radius * t * np.cos(base_angle),
                radius * t * np.sin(base_angle),
                0
            ])
            poses.append((forward_rotation, pos))

        # Smooth rotation to the next base angle
        if rotation_idx < num_rotations - 1:
            next_base_angle = (rotation_idx + 1) * delta_angle * (np.pi / 180)
            next_z_axis = (np.cos(next_base_angle), np.sin(next_base_angle), 0)
            next_x_axis = np.cross(y_axis, next_z_axis)
            next_forward_rotation = np.stack([next_x_axis, y_axis, next_z_axis], axis=1)

            # Calculate the angle difference
            angle_diff = np.degrees(np.arccos(np.clip(np.dot(z_axis, next_z_axis), -1.0, 1.0)))
            
            # Determine the number of interpolation steps based on angle difference
            max_angle_diff = 2  # degrees
            num_interpolations = max(1, int(np.ceil(angle_diff / max_angle_diff)))

            # Interpolate between current and next rotation
            for interp_idx in range(num_interpolations + 1):
                interp_t = interp_idx / num_interpolations
                interpolated_rotation = (1 - interp_t) * forward_rotation + interp_t * next_forward_rotation
                poses.append((interpolated_rotation, pos))


    return poses


def create_curved_trajectory_with_poses_forward_backward_old(depth, filter_max_depth=100):
    radius = calculate_radius_from_depth(depth, filter_max_depth=filter_max_depth)
    print(f"radius: {radius}")
    
    poses = []
    num_intermediate_points = 5  # Points for smooth transition
    num_rotations = int(360 / 45)  # Number of different angles (360/45 = 8)
    
    for rotation_idx in range(num_rotations):
        base_angle = rotation_idx * 45 * (np.pi / 180)
        
        # Forward movement - camera faces forward
        y_axis = (0, 0, -1)
        z_axis = (np.cos(base_angle), np.sin(base_angle), 0)
        x_axis = np.cross(y_axis, z_axis)
        forward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        for j in range(num_intermediate_points + 1):
            t = j / num_intermediate_points
            pos = np.array([
                radius * t * np.cos(base_angle),
                radius * t * np.sin(base_angle),
                0
            ])
            poses.append((forward_rotation, pos))
        
        # Backward movement - camera faces backward
        z_axis = (-np.cos(base_angle), -np.sin(base_angle), 0)  # Reverse direction
        x_axis = np.cross(y_axis, z_axis)
        backward_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        
        for j in range(num_intermediate_points, -1, -1):
            t = j / num_intermediate_points
            pos = np.array([
                radius * t * np.cos(base_angle),
                radius * t * np.sin(base_angle),
                0
            ])
            poses.append((backward_rotation, pos))

    return poses


def create_curved_trajectory_with_poses_circle(depth, num_points, filter_max_depth=100):
    radius = calculate_radius_from_depth(depth, filter_max_depth=filter_max_depth)

    print(f"radius: {radius}")
    
    poses = []
    y_axis = (0, 0, -1)
    z_axis = (1, 0, 0)
    x_axis = np.cross(y_axis, z_axis)
    first_rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)  # R_c2w
    
    num_intermediate_points = 5  # Number of intermediate points
    for j in range(num_intermediate_points):
        interp_x = radius * j/num_intermediate_points
        interp_y = 0
        interp_z = 0
        poses.append((first_rotation_matrix, np.array([interp_x, interp_y, interp_z])))

    print(f"poses[0]: {poses[0]}")

    for i in range(num_points):
        angle = (i / num_points) * 2 * np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0

        y_axis = (0, 0, -1)
        z_axis = (np.cos(angle), np.sin(angle), 0)
        x_axis = np.cross(y_axis, z_axis)

        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)  # R_c2w
        translation_vector = np.array([x, y, z])  # t_c2w
        poses.append((rotation_matrix, translation_vector))

    return poses

def display_point_cloud_with_poses(point_cloud=None, poses=None):
    geometries = []

    if point_cloud is not None:
        geometries.append(point_cloud)

    if poses is not None:
        for r_c2w_cv, t_c2w_cv in poses:
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=t_c2w_cv)
            camera.rotate(r_c2w_cv, center=t_c2w_cv)
            geometries.append(camera)
        trajectory_points = [pose[1] for pose in poses]
        trajectory_lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
        trajectory_colors = [[1, 0, 0] for _ in range(len(trajectory_lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory_points),
            lines=o3d.utility.Vector2iVector(trajectory_lines),)
        line_set.colors = o3d.utility.Vector3dVector(trajectory_colors)
        geometries.append(line_set)

    # 使用线程来显示几何体
    threading.Thread(target=o3d.visualization.draw_geometries, args=(geometries,)).start()  # 修改此行

def warp_affine(poses, first_wide_img, first_wide_depth, focus, render_focus, render_h, render_w, output_dir="output"):
    rgb_out_dir = os.path.join(output_dir, "rgb_warp")
    if not os.path.exists(rgb_out_dir):
        os.makedirs(rgb_out_dir)

    base_angle = np.deg2rad(90.0)
    y_axis = (0, 0, -1)
    z_axis = (np.cos(base_angle), np.sin(base_angle), 0)
    x_axis = np.cross(y_axis, z_axis)
    global_camera_R = np.stack([x_axis, y_axis, z_axis], axis=1)

    cx = first_wide_img.shape[1] // 2
    cy = first_wide_img.shape[0] // 2
    K1 = np.array([
        [focus, 0, cx],
        [0, focus, cy],
        [0, 0, 1]
    ])
    K2 = np.array([
        [render_focus, 0, render_w//2],
        [0, render_focus, render_h//2],
        [0, 0, 1]
    ])
    dsize = (render_w, render_h)
    warp_imgs = []
    warp_poses = []
    for idx, pose in enumerate(poses):
        rotation_matrix = pose[0]
        R = rotation_matrix.T @ global_camera_R
        H = K2 @ R @ np.linalg.inv(K1)  
        img_np = cv2.warpPerspective(first_wide_img, H, dsize, flags=cv2.INTER_AREA)
        warp_imgs.append(img_np)
        warp_pose = (pose[0], np.array([0, 0, 0])) 
        warp_poses.append(warp_pose)
        # crop 480x720
        if render_h != img_np.shape[0] or render_w != img_np.shape[1]:
            start_h = (img_np.shape[0] - render_h) // 2
            start_w = (img_np.shape[1] - render_w) // 2
            img_np = img_np[start_h:start_h+render_h, start_w:start_w+render_w, :]
        cv2.imwrite(os.path.join(rgb_out_dir, f"{idx}.png"), img_np)
    return warp_imgs, warp_poses

def render_point_cloud_with_gs(poses, first_wide_img, first_wide_depth, focus, render_focus, render_h, render_w, warp_imgs, warp_poses, use_gs_opt=False, output_dir="output"):
    scene = Scene(args)
    scene.init_from_rgbd(first_wide_img, first_wide_depth, focus, render_focus, warp_imgs, warp_poses, output_dir, use_gs_opt=use_gs_opt)
    # scene.init_from_rgbd(first_wide_img, first_wide_depth, focus,out_path=output_dir)

    fovx = focal2fov(render_focus, render_w)
    fovy = focal2fov(render_focus, render_h)
    print(f"render fovx: {fovx/np.pi*180}, render fovy: {fovy/np.pi*180}")

    rgb_out_dir = os.path.join(output_dir, "rgb")
    if not os.path.exists(rgb_out_dir):
        os.makedirs(rgb_out_dir)
    imgs = []
    for idx, pose in enumerate(poses):
        cur_pose = np.eye(4)
        cur_pose[:3,:3] = pose[0]
        cur_pose[:3,3] = pose[1]
        cur_camera = Camera(np.linalg.inv(cur_pose), fovx, fovy, render_w, render_h)
        out = scene.render(cur_camera)
        img = out['render']
        img_np = (img.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
        if img_np.shape[0] != render_h or img_np.shape[1] != render_w:
            start_h = (img_np.shape[0] - render_h) // 2
            start_w = (img_np.shape[1] - render_w) // 2
            crop_img = img_np[start_h:start_h+render_h, start_w:start_w+render_w, :]
            imgs.append(crop_img)
            cv2.imwrite(os.path.join(rgb_out_dir, f"{idx}.png"), crop_img)
        else:
            cv2.imwrite(os.path.join(rgb_out_dir, f"{idx}.png"), img_np)
            imgs.append(img_np)

    imgs = np.stack(imgs, axis=0)
    array_to_video(imgs, os.path.join(output_dir, 'rgb_render.mp4'), fps=8)

def create_point_cloud(bgr, depth, filter_max_depth=100.0, fovx=None, filename="cloud.ply"):
    print(f"depth.shape: {depth.shape}")
    print(f"depth.max(): {depth.max()}, min: {depth.min()}")
    height, width = depth.shape
    if fovx is None:
        raise ValueError("focal_length is required for wide_flag")
    focus = width / 2 / np.tan(fovx/2 * np.pi / 180)
    fovy = np.arctan(height / 2 / focus) * 2 * 180 / np.pi
    print(f"original_fovx: {fovx}, originalfovy: {fovy}, original_focus: {focus}")
    # Create a meshgrid for x and y
    x, y = np.meshgrid(np.arange(width), np.arange(height))
            
    # Calculate the 3D coordinates from depth
    z = depth.flatten()
    x_coords = (x.flatten() - width / 2) * z / focus  # Assuming focal_length is defined
    y_coords = (y.flatten() - height / 2) * z / focus  # Invert y-axis for image coordinates
            
    # Create a mask for valid depth values
    valid_mask = (z >0) & (z <filter_max_depth)
            
    # Filter the coordinates and colors based on the valid mask
    # points = np.column_stack((x_coords[valid_mask], y_coords[valid_mask], z[valid_mask]))
    points = np.column_stack((x_coords[valid_mask], z[valid_mask], -y_coords[valid_mask]))
            
    colors = bgr.reshape(-1, 3)[valid_mask] / 255.0  # Normalize colors to [0, 1]
    colors = colors[..., [2,1,0]]  # Convert BGR to RGB
    radius = z[valid_mask] / focus
    print("cloud radius: ", radius)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save point cloud to PLY file
    o3d.io.write_point_cloud(filename, point_cloud)  # Specify your desired file name and path

    return point_cloud, radius, focus


def extract_index(filename):
    """提取文件名中的数字索引"""
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

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

import argparse 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process panorama and depth images.")
    parser.add_argument('--rgb_dir', type=str, default="./test_gen_0114/re10k_first_gen/", help='Path to the panorama images directory')
    parser.add_argument('--depth_dir', type=str, default="./test_gen_0114/re10k_first_gen_depth/", help='Path to the depth images directory')
    parser.add_argument('--pose_dir', type=str, default="./test_gen_0114/re10k_first_gen_poses/", help='Path to the pose files directory')
    parser.add_argument('--output_path', type=str, default="./output_wide_test", help='Path to the output directory')
    args = parser.parse_args()
    rgb_dir = args.rgb_dir
    depth_dir = args.depth_dir
    pose_dir = args.pose_dir
    output_path = args.output_path

    data_type = "png"  # exr

    render_h = 480
    render_w = 720
    original_fovx = 120
    debug_flag = False
    num_circle = 3
    use_gs_opt = False

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')], key=extract_index)
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')], key=extract_index)

    for rgb_file, depth_file in zip(rgb_files, depth_files):
        # idx = int(rgb_file.split("_")[-1].split(".")[0])
        idx = extract_index(rgb_file)
        print("idx=== ", idx, rgb_file, depth_file)

        if idx !=1:
            continue

        cur_output_path = os.path.join(output_path, f"{idx}")
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)

        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_exr_path = os.path.join(depth_dir, depth_file)

        rgb, depth = load_images(rgb_path, depth_exr_path, data_type=data_type)
        point_cloud, radius, focus = create_point_cloud(rgb, depth, filter_max_depth=100, fovx=original_fovx)

        render_focus = focus * 0.7
        render_fovx = focal2fov(render_focus, render_w)/np.pi*180
        render_fovy = focal2fov(render_focus, render_h)/np.pi*180
        print(f"render_fovx: {render_fovx}, render_fovy: {render_fovy}, render_focus: {render_focus}")
        # poses = create_curved_trajectory_with_poses_forward_backward_circle_wide(depth, delta_fov=(original_fovx-render_fovx), num_circle=num_circle, two_layer_flag=False)
        # poses = create_curved_trajectory_with_poses_forward_backward_rotation(depth, delta=60)
        poses = read_pose_matrix(os.path.join(pose_dir, f"{idx}.txt"))

        if debug_flag:
            display_point_cloud_with_poses(point_cloud, poses)

        # warp_imgs, warp_poses = warp_affine(poses, rgb, depth, focus, render_focus, render_h, render_w, output_dir=cur_output_path)
        # render_point_cloud_with_gs(poses, rgb, depth, focus, render_focus, render_h, render_w, warp_imgs, warp_poses, use_gs_opt=use_gs_opt, output_dir=cur_output_path)

        # replace_start_end_frames_by_warp_imgs(num_circle=num_circle, output_dir=cur_output_path, num_images=46)

