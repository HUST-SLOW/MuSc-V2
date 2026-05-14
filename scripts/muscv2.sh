device=0
# MVTec 3D-AD dataset
# 2D modal
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py --device 0 \
--data_path ./data/mvtec_3d/ --dataset_name mvtec_3d --class_name ALL \
--backbone_name dino_vitbase8 --feature_layers 3 7 11 \
--img_resize 224 --r_list 1 3 5 --output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm

# 3D modal
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py --device 0 \
--data_path ./data/mvtec_3d/ --dataset_name mvtec_3d --class_name ALL \
--backbone_name point-mae --feature_layers 3 7 11 \
--img_resize 224 --r_list 1 3 5 --output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm

# 2D+3D modal
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py --device 0 \
--data_path ./data/mvtec_3d/ --dataset_name mvtec_3d --class_name ALL \
--backbone_name dino_vitbase8 point-mae --feature_layers 3 7 11 \
--img_resize 224 --r_list 1 3 5 --output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm

# Eyecandies dataset
# 2D modal
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py --device 0 \
--data_path ./data/eyecandies/ --dataset_name eyecandies --class_name ALL \
--backbone_name dino_vitbase8 --feature_layers 3 7 11 \
--img_resize 224 --r_list 1 3 5 --output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm

# 3D modal
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py --device 0 \
--data_path ./data/eyecandies/ --dataset_name eyecandies --class_name ALL \
--backbone_name point-mae --feature_layers 3 7 11 \
--img_resize 224 --r_list 1 3 5 --output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm

# 2D+3D modal
CUDA_VISIBLE_DEVICES=${device} python examples/muscv2_main.py --device 0 \
--data_path ./data/eyecandies/ --dataset_name eyecandies --class_name ALL \
--backbone_name dino_vitbase8 point-mae --feature_layers 3 7 11 \
--img_resize 224 --r_list 1 3 5 --output_dir ./output/ \
--vis True --save_excel True --vis_type whole_norm
