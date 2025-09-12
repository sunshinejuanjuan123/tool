import numpy as np
import imageio
from PIL import Image
import matplotlib
import open3d as o3d
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
    
depth_knn=np.asarray(imageio.imread('pts_depth.png')).astype(np.float32)/1000
depth_vggt=np.asarray(imageio.imread('/data1/0000.png')).astype(np.float32)/1000
depth_vggt_normal = (depth_vggt-depth_vggt.min())/ (depth_vggt.max()-depth_vggt.min())
depth_knn_min = depth_knn.min()
depth_knn_max = depth_knn.max()
depth_knn_normal = (depth_knn-depth_knn_min)/ (depth_knn_max-depth_knn_min)

depth_vggt_normal2gt = depth_vggt_normal*(depth_knn_max-depth_knn_min)+depth_knn_min

depth_vggt_normal2gt_int = (depth_vggt_normal2gt*1000).astype(np.uint16)
Image.fromarray(depth_vggt_normal2gt_int).save('depth_vggt_normal2gt.png')



depth_vggt_normal2gt_vis = visualize_depth(depth_vggt_normal2gt,
                                           depth_min=depth_knn_min,
                                           depth_max=depth_knn_max)
Image.fromarray(depth_vggt_normal2gt_vis).save('depth_vggt_normal2gt_vis.png')




# In [10]: depth_knn_max
# Out[10]: 65.535

# In [11]: depth_knn_min
# Out[11]: 8.179
depth_knn_max=65.535
depth_knn_min=8.179

scale=1/1000
pcd=o3d.io.read_point_cloud('/data1/experienments/vggt_output/center_camera_fov30_3/ply/world_points_0.ply')
pcd_scale=pcd.scale(float(scale),center=np.array([0,0,0]))
# o3d.io.write_point_cloud('0000_scale.ply',pcd_scale)
points = np.asarray(pcd_scale.points)
z=points[:,2]
knn_scale = (depth_knn_max - depth_knn_min)/(z.max()-z.min())
pcd_knn_scale=pcd.scale(float(knn_scale),center=np.array([0,0,0]))
o3d.io.write_point_cloud('0000_knn_scale.ply',pcd_knn_scale)

pcd_knn_scale_points = np.asarray(pcd_knn_scale.points)


