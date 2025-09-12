source activate /iag_ad_01/ad/yuanweizhong/miniconda/streetcrafter

cd /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data_processor/sensetime_processor

python save_c2w.py --raw_scene_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser \
--save_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser/camera/example_dataset/example_scene/poses/