import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path
from PIL import Image
import imageio.v3 as iio
import yaml
from tqdm import tqdm

# The same camera has been used for all the images
FOCAL_LENGTH = 711.11

def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',
    ]

def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg

def depth_to_pointcloud(depth_img, info_depth, pose_txt, focal_length):
    # input depth map (in meters) --- cfr previous section
    depth_mt = load_and_convert_depth(depth_img, info_depth)    # (512, 512)

    # input pose
    pose = np.loadtxt(pose_txt)

    # camera intrinsics
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

    # build the camera projection matrix
    camera_proj = intrinsics_4x4 @ pose

    # build the (u, v, 1, 1/depth) vectors (non optimized version)
    camera_vectors = np.zeros((width * height, 4))
    count=0
    for j in range(height):
        for i in range(width):
            camera_vectors[count, :] = np.array([i, j, 1, 1/depth_mt[j, i]])
            count += 1

    # invert and apply to each 4-vector
    hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.T

    # remove the homogeneous coordinate
    pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T
    return pcd[:, :3]


def remove_plane(unorganized_pc, path, distance = 0.01):
    special_classnames = ['GummyBear', 'LicoriceSandwich', 'Marshmallow']
    flag = 0
    for c in special_classnames:
        if c in path:
            flag = 1
            break
    if flag == 0:
        depth = unorganized_pc[:, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        plane_idx = depth >= depth.max() - distance
    else:
        depth = unorganized_pc[:, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        plane_idx = depth > depth.max()
    planeless_pc = unorganized_pc.copy()
    planeless_pc[plane_idx] = 0

    return planeless_pc


def connected_components_cleaning(unorganized_pc, light_image_path):
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

    rgb = Image.open(light_image_path).convert('RGB')
    rgb = np.array(rgb)
    rgb = rgb.reshape(-1, 3)

    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    labels = np.array(o3d_pc.cluster_dbscan(eps=0.03, min_points=30, print_progress=False))

    unique_cluster_ids, cluster_size = np.unique(labels,return_counts=True)
    largest_cluster_id = -1
    brightest_rgb = 0
    for id in unique_cluster_ids[1:]:
        outlier_indices_nonzero_array = np.argwhere(labels == id)
        outlier_indices_original_pc_array = nonzero_indices[outlier_indices_nonzero_array]
        rgb_max = rgb[outlier_indices_original_pc_array].max()
        if rgb_max > brightest_rgb:
            largest_cluster_id = id
            brightest_rgb = rgb_max

    outlier_indices_nonzero_array = np.argwhere(labels != largest_cluster_id)
    outlier_indices_original_pc_array = nonzero_indices[outlier_indices_nonzero_array]
    unorganized_pc[outlier_indices_original_pc_array] = 0

    return unorganized_pc

def preprocess_pc(depth_map_path):
    # READ FILES
    dimg = iio.imread(depth_map_path)
    dimg = dimg.astype(np.float32)
    h, w = dimg.shape
    info_depth_path = str(depth_map_path).replace("depth.png", "info_depth.yaml")
    pose_path = str(depth_map_path).replace("depth.png", "pose.txt")
    image_path = str(depth_map_path).replace("depth", "image_0")
    # depth map to point clouds
    # (N, 3)
    pc = depth_to_pointcloud(
        depth_map_path,
        info_depth_path,
        pose_path,
        FOCAL_LENGTH,
    )
    # REMOVE PLANE
    planeless_unorganized_pc = remove_plane(pc, str(depth_map_path))
    planeless_unorganized_pc = connected_components_cleaning(planeless_unorganized_pc, image_path)

    # save xyz map and foreground mask
    planeless_organized_pc = planeless_unorganized_pc.reshape(h, w, -1)
    planeless_organized_pc = planeless_organized_pc.astype(np.float32)
    xyz_map_path = str(depth_map_path).replace("depth.png", "xyz.tiff")
    tiff.imwrite(xyz_map_path, planeless_organized_pc)
    nonzero_indices = np.nonzero(np.all(planeless_organized_pc != 0, axis=2))
    foreground_mask = np.zeros((h, w))
    foreground_mask[nonzero_indices[0], nonzero_indices[1]] = 255
    foreground_mask = foreground_mask.astype(np.uint8)
    foreground_mask_path = str(depth_map_path).replace("depth", "foreground")
    Image.fromarray(foreground_mask).save(foreground_mask_path)

if __name__ == '__main__':
    root_path = '/data/eyecandies/'
    paths = Path(root_path).rglob('*depth.png')
    print('Processing...')
    processed_classname = []
    processed_files = 0
    exist_classname = []
    for path in tqdm(paths):
        class_name = str(path).split('/')[-4]
        if class_name in exist_classname:
            continue
        if class_name not in processed_classname:
            processed_classname.append(class_name)
            print('start processing {}...'.format(class_name))
        preprocess_pc(path)
        if processed_files % 100 == 0:
            print(f"Processed {processed_files} files...")
        processed_files += 1


