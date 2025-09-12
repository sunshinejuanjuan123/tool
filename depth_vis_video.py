import numpy as np
import imageio
from PIL import Image
import matplotlib
import os
from tqdm import tqdm
import cv2
def visualize_depth(depth: np.ndarray, 
                    depth_min=None, 
                    depth_max=None, 
                    percentile=2, 
                    ret_minmax=False,
                    cmap='Spectral'):
    if depth_min is None: depth_min = np.percentile(depth, percentile)
    if depth_max is None: depth_max = np.percentile(depth, 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np
# In [10]: depth_knn_max
# Out[10]: 65.535

# In [11]: depth_knn_min
# Out[11]: 8.179

depth_path='/data1/experienments/vggt_output/center_camera_fov30_3/depth'
depth_vis_path='/data1/experienments/vggt_output/center_camera_fov30_3/depth_vis'
depth_vis_color_path='/data1/experienments/vggt_output/center_camera_fov30_3/depth_vis_color'
if not os.path.exists(depth_vis_path):
    os.makedirs(depth_vis_path)
if not os.path.exists(depth_vis_color_path):
    os.makedirs(depth_vis_color_path)
depth_knn_max=65.535
depth_knn_min=8.179
#存视频
fps=10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width=3840
height=2160
frame_size = (width, height)
file_name='depth_vis.mp4'
video_writer = cv2.VideoWriter(file_name, fourcc, fps, frame_size,isColor=True)

for depth_file in tqdm(sorted(os.listdir(depth_path)),desc='depth_vis'):
    depth_vggt=np.asarray(imageio.imread(os.path.join(depth_path,depth_file))).astype(np.float32)/1000
    depth_vggt_normal = (depth_vggt-depth_vggt.min())/ (depth_vggt.max()-depth_vggt.min())
    depth_vggt_normal2gt = depth_vggt_normal*(depth_knn_max-depth_knn_min)+depth_knn_min
    depth_vggt_normal2gt_int = (depth_vggt_normal2gt*1000).astype(np.uint16)
    # Image.fromarray(depth_vggt_normal2gt_int).save(os.path.join(depth_vis_path,depth_file))
    depth_vggt_normal2gt_vis = visualize_depth(depth_vggt_normal2gt,
                                                depth_min=depth_knn_min,
                                                depth_max=depth_knn_max)
    # Image.fromarray(depth_vggt_normal2gt_vis).save(os.path.join(depth_vis_color_path,depth_file))
    video_writer.write(depth_vggt_normal2gt_vis[:,:,::-1])
    print(f'{depth_file} shape: {depth_vggt_normal2gt_vis.shape}')
    # cv2.imshow('Recording', depth_vggt_normal2gt_vis)
    # cv2.imwrite(os.path.join('test.png'), depth_vggt_normal2gt_vis)
    # cv2.waitKey(1)
# import ipdb;ipdb.set_trace()
video_writer.release()
