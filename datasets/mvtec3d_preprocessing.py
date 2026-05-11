import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path
from PIL import Image
import math
import utils.tools as mvt_util

def get_edges_of_pc(organized_pc):
    unorganized_edges_pc = organized_pc[0:10, :, :].reshape(organized_pc[0:10, :, :].shape[0]*organized_pc[0:10, :, :].shape[1],organized_pc[0:10, :, :].shape[2])
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc,organized_pc[-10:, :, :].reshape(organized_pc[-10:, :, :].shape[0] * organized_pc[-10:, :, :].shape[1],organized_pc[-10:, :, :].shape[2])],axis=0)
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, 0:10, :].reshape(organized_pc[:, 0:10, :].shape[0] * organized_pc[:, 0:10, :].shape[1],organized_pc[:, 0:10, :].shape[2])], axis=0)
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, -10:, :].reshape(organized_pc[:, -10:, :].shape[0] * organized_pc[:, -10:, :].shape[1],organized_pc[:, -10:, :].shape[2])], axis=0)
    unorganized_edges_pc = unorganized_edges_pc[np.nonzero(np.all(unorganized_edges_pc != 0, axis=1))[0],:]
    return unorganized_edges_pc

def get_plane_eq(unorganized_pc,ransac_n_pts=50):
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.004, ransac_n=ransac_n_pts, num_iterations=1000)
    return plane_model

def remove_plane(organized_pc_clean, organized_rgb ,distance_threshold=0.005):
    # PREP PC
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc_clean)
    unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)
    clean_planeless_unorganized_pc = unorganized_pc.copy()
    planeless_unorganized_rgb = unorganized_rgb.copy()

    # REMOVE PLANE
    plane_model = get_plane_eq(get_edges_of_pc(organized_pc_clean))
    distances = np.abs(np.dot(np.array(plane_model), np.hstack((clean_planeless_unorganized_pc, np.ones((clean_planeless_unorganized_pc.shape[0], 1)))).T))
    plane_indices = np.argwhere(distances < distance_threshold)

    planeless_unorganized_rgb[plane_indices] = 0
    clean_planeless_unorganized_pc[plane_indices] = 0
    clean_planeless_organized_pc = clean_planeless_unorganized_pc.reshape(organized_pc_clean.shape[0],
                                                                          organized_pc_clean.shape[1],
                                                                          organized_pc_clean.shape[2])
    planeless_organized_rgb = planeless_unorganized_rgb.reshape(organized_rgb.shape[0],
                                                                          organized_rgb.shape[1],
                                                                          organized_rgb.shape[2])
    return clean_planeless_organized_pc, planeless_organized_rgb

def connected_components_cleaning(organized_pc, organized_rgb, image_path = ''):
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc)
    unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)

    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))


    unique_cluster_ids, cluster_size = np.unique(labels,return_counts=True)
    max_label = labels.max()
    if max_label>0:
        print("##########################################################################")
        print(f"Point cloud file {image_path} has {max_label + 1} clusters")
        print(f"Cluster ids: {unique_cluster_ids}. Cluster size {cluster_size}")
        print("##########################################################################\n\n")

    largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]
    outlier_indices_nonzero_array = np.argwhere(labels != largest_cluster_id)
    outlier_indices_original_pc_array = nonzero_indices[outlier_indices_nonzero_array]
    unorganized_pc[outlier_indices_original_pc_array] = 0
    unorganized_rgb[outlier_indices_original_pc_array] = 0
    organized_clustered_pc = unorganized_pc.reshape(organized_pc.shape[0],
                                                                          organized_pc.shape[1],
                                                                          organized_pc.shape[2])
    organized_clustered_rgb = unorganized_rgb.reshape(organized_rgb.shape[0],
                                                    organized_rgb.shape[1],
                                                    organized_rgb.shape[2])
    return organized_clustered_pc, organized_clustered_rgb

def roundup_next_100(x):
    return int(math.ceil(x / 100.0)) * 100

def pad_cropped_pc(cropped_pc, single_channel=False):
    orig_h, orig_w = cropped_pc.shape[0], cropped_pc.shape[1]
    round_orig_h = roundup_next_100(orig_h)
    round_orig_w = roundup_next_100(orig_w)
    large_side = max(round_orig_h, round_orig_w)

    a = (large_side - orig_h) // 2
    aa = large_side - a - orig_h

    b = (large_side - orig_w) // 2
    bb = large_side - b - orig_w
    if single_channel:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb)), mode='constant')
    else:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

def preprocess_test(organized_rgb, organized_pc):
    # REMOVE PLANE
    planeless_organized_pc, planeless_organized_rgb = remove_plane(organized_pc, organized_rgb)

    # PAD WITH ZEROS TO LARGEST SIDE (SO THAT THE FINAL IMAGE IS SQUARE)
    padded_planeless_organized_pc = pad_cropped_pc(planeless_organized_pc, single_channel=False)
    padded_planeless_organized_rgb = pad_cropped_pc(planeless_organized_rgb, single_channel=False)

    organized_clustered_pc, organized_clustered_rgb = connected_components_cleaning(padded_planeless_organized_pc,
                                                                                    padded_planeless_organized_rgb)
    return organized_clustered_rgb, organized_clustered_pc

def preprocess_pc(tiff_path):
    # READ FILES
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
    organized_rgb = np.array(Image.open(rgb_path))

    organized_gt = None
    gt_exists = os.path.isfile(gt_path)
    if gt_exists:
        organized_gt = np.array(Image.open(gt_path))

    # REMOVE PLANE
    planeless_organized_pc, planeless_organized_rgb = remove_plane(organized_pc, organized_rgb)


    # PAD WITH ZEROS TO LARGEST SIDE (SO THAT THE FINAL IMAGE IS SQUARE)
    padded_planeless_organized_pc = pad_cropped_pc(planeless_organized_pc, single_channel=False)
    padded_planeless_organized_rgb = pad_cropped_pc(planeless_organized_rgb, single_channel=False)
    if gt_exists:
        padded_organized_gt = pad_cropped_pc(organized_gt, single_channel=True)

    organized_clustered_pc, organized_clustered_rgb = connected_components_cleaning(padded_planeless_organized_pc, padded_planeless_organized_rgb, tiff_path)
    # SAVE PREPROCESSED FILES
    tiff.imsave(tiff_path, organized_clustered_pc)
    Image.fromarray(organized_clustered_rgb).save(rgb_path)
    if gt_exists:
        Image.fromarray(padded_organized_gt).save(gt_path)

if __name__ == '__main__':
    root_path = './data/mvtec_3d/'
    paths = Path(root_path).rglob('*.tiff')
    print(f"Found {len(list(paths))} tiff files in {root_path}")
    processed_files = 0
    for path in Path(root_path).rglob('*.tiff'):
        preprocess_pc(path)
        processed_files += 1
        if processed_files % 50 == 0:
            print(f"Processed {processed_files} tiff files...")









