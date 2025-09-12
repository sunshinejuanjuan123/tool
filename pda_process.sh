# 1.运行lidar2image.py 
# root_path = '/data/senseauto/高速远距离数据/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser/'
# 设置为入参
# 2.运行knn_insert.py
# root_path = '/data/senseauto/高速远距离数据/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser/'
# 设置为入参
python lidar2image.py --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser
python knn_insert.py --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser

