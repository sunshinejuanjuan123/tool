import cv2
import os
import numpy as np
from tqdm import tqdm
str_shift='_shift_3.00'
# str_shift='_shift_2.00'
def make_side_by_side_video(img_dir1, img_dir2, output_path,
                            fps=25, codec='mp4v', size=None):
    """
    将两个文件夹中的图片按序左右拼接，并写入视频。

    Args:
        img_dir1 (str): 第一个图片序列所在文件夹路径，图片按文件名排序。
        img_dir2 (str): 第二个图片序列所在文件夹路径，图片按文件名排序。
        output_path (str): 输出视频文件路径（含扩展名，如 .mp4）。
        fps (int): 帧率，默认 25。
        codec (str): 视频编码（四字符编码），默认 'mp4v'。
        size (tuple or None): 输出视频的 (width, height)。若为 None，则使用第一个拼接帧的尺寸。
    """
    # 获取并排序文件列表
    imgs1 = sorted([os.path.join(img_dir1, f) for f in os.listdir(img_dir1)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask.png' not in f])
    imgs2 = sorted([os.path.join(img_dir2, f) for f in os.listdir(img_dir2)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and str_shift in f])
    # import ipdb;ipdb.set_trace()
    # 确保两序列长度相同；若不同，可取最小长度或其他策略
    n_frames = min(len(imgs1), len(imgs2))
    if n_frames == 0:
        raise ValueError("至少需要每个文件夹各一张图片。")

    # 读取第一帧，确定输出尺寸
    img1 = cv2.imread(imgs1[0])
    img2 = cv2.imread(imgs2[0])
    # 如果两张图尺寸不同，可以先缩放到同一高度
    h = min(img1.shape[0], img2.shape[0])
    w1 = int(img1.shape[1] * (h / img1.shape[0]))
    w2 = int(img2.shape[1] * (h / img2.shape[0]))
    img1 = cv2.resize(img1, (w1, h))
    img2 = cv2.resize(img2, (w2, h))

    # 拼接后宽高
    concat_w = w1 + w2
    concat_h = h

    # 如果用户指定了输出大小，则覆盖
    if size is not None:
        concat_w, concat_h = size

    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (concat_w, concat_h))

    for i in range(n_frames):
        img1 = cv2.imread(imgs1[i])
        img2 = cv2.imread(imgs2[i])
        # 缩放到相同高度
        img1 = cv2.resize(img1, (w1, h))
        img2 = cv2.resize(img2, (w2, h))
        # 拼接
        concat = np.hstack((img1, img2))
        # 如果需要适配到输出尺寸
        if size is not None:
            concat = cv2.resize(concat, (concat_w, concat_h))
        writer.write(concat)

    writer.release()
    print(f"视频已保存到 {output_path}")

if __name__ == "__main__":
    # 示例调用
    #pointrender
    folder1 = "/data/senseauto/176/lidar//color_render" + str_shift
    #diffusion
    folder2 = "/data/senseauto/waymo_val_176/diffusion/"
    output_video = "side_by_side.mp4"
    make_side_by_side_video(folder1, folder2, output_video,
                            fps=30, codec='mp4v')
