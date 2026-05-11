import torch
from tqdm import tqdm


def patch2point(patch_num, nonzero_indices_center_list, image_size):
    # (B, L, C) (B, N) (B, N, group_size)
    patch_num_ = int(patch_num**0.5)
    patch_size = int(image_size // patch_num_)
    patch2point_list = []
    for img_idx in range(nonzero_indices_center_list.shape[0]):
        patch2point_img_list = []
        point_idx = nonzero_indices_center_list[img_idx]
        for patch_idx in range(patch_num):
            patch_coors = (patch_idx//patch_num_, patch_idx%patch_num_)
            pixel_coors = (patch_coors[0]*patch_size, patch_coors[1]*patch_size)
            point_idx_ = torch.zeros_like(point_idx)
            point_idx_[point_idx//image_size<pixel_coors[0]] = -1
            point_idx_[point_idx//image_size>=pixel_coors[0]+patch_size] = -1
            point_idx_[point_idx%image_size<pixel_coors[1]] = -1
            point_idx_[point_idx%image_size>=pixel_coors[1]+patch_size] = -1
            point_idx_ = torch.where(point_idx_ != -1)
            patch2point_img_list.append(point_idx_[0])
        patch2point_list.append(patch2point_img_list)
    return patch2point_list


def point2patch(patch_num, nonzero_indices_center_list, image_size):
    # (B, L, C) (B, N)
    patch_num_ = int(patch_num**0.5)
    patch_size = int(image_size // patch_num_)
    point_coor = (nonzero_indices_center_list//image_size, nonzero_indices_center_list%image_size)
    patch_coor = (point_coor[0]//patch_size, point_coor[1]//patch_size)
    patch_idx = patch_coor[0]*patch_num_+patch_coor[1]
    point2patch_list = []
    for img_idx in range(patch_idx.shape[0]):
        point2patch_img_list = []
        for point_idx in range(patch_idx.shape[1]):
            patch_idx_ = torch.unique(patch_idx[img_idx, point_idx])
            point2patch_img_list.append(patch_idx_)
        point2patch_list.append(point2patch_img_list)
    return point2patch_list


def cross_modal_align(Z_org, Z_cross, proj_index):
    # cross-modal alignment
    image_num, patch_num, _ = Z_org.shape
    _, _, c_cross = Z_cross.shape
    Z_org_cross = torch.zeros((image_num, patch_num, c_cross)).to(Z_org.device)
    for img_idx in range(image_num):
        for patch_idx in range(patch_num):
            patch_proj_cross_idx = proj_index[img_idx][patch_idx]
            if patch_proj_cross_idx.shape[0]==0:
                continue
            patch_proj_cross_features = Z_cross[img_idx][patch_proj_cross_idx]
            Z_org_cross[img_idx, patch_idx] = patch_proj_cross_features.mean(0)
    return Z_org_cross


def replace_sorted(source, target):
    _, indices_sourse = torch.sort(source, dim=-1)
    _, indices_target = torch.sort(target, dim=-1)
    indices_target_inv = torch.argsort(indices_target, dim=-1)
    indices_to_replace = torch.gather(indices_sourse, dim=-1, index=indices_target_inv)
    replaced_target = torch.gather(source, dim=-1, index=indices_to_replace)
    return replaced_target


def compute_scores_single(Z, i, topmin_max=0.3):
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    ref_image_num = Z_ref.shape[0]
    # L2 distance
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, ref_image_num, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_max = int(patch2image.shape[1]*topmin_max)
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)


def compute_scores_multiple(Z_2d, Z_3d, i, topmin_max=0.3, patch2point_list=None, point2patch_list=None):
    # compute anomaly scores
    image_num, patch_num_2d, c_2d = Z_2d.shape
    _, patch_num_3d, c_3d = Z_3d.shape
    Z_2d_ref = torch.cat((Z_2d[:i], Z_2d[i+1:]), dim=0)
    Z_3d_ref = torch.cat((Z_3d[:i], Z_3d[i+1:]), dim=0)
    ref_image_num = Z_2d_ref.shape[0]
    # L2 distance
    patch2image_2d_org = torch.cdist(Z_2d[i:i+1], Z_2d_ref.reshape(-1, c_2d)).reshape(patch_num_2d, ref_image_num, patch_num_2d)
    patch2image_3d_org = torch.cdist(Z_3d[i:i+1], Z_3d_ref.reshape(-1, c_3d)).reshape(patch_num_3d, ref_image_num, patch_num_3d)
    patch2image_2d_org, _ = torch.min(patch2image_2d_org, -1)
    patch2image_3d_org, _ = torch.min(patch2image_3d_org, -1)
    patch2image_2d_cross = cross_modal_align(patch2image_2d_org.unsqueeze(0), patch2image_3d_org.unsqueeze(0), patch2point_list[i:i+1])[0]
    patch2image_3d_cross = cross_modal_align(patch2image_3d_org.unsqueeze(0), patch2image_2d_org.unsqueeze(0), point2patch_list[i:i+1])[0]
    # calculate the confidence weight
    confidence_2d = 1 - replace_sorted(patch2image_2d_org.std(-1), patch2image_2d_cross.std(-1))
    confidence_3d = 1 - replace_sorted(patch2image_3d_org.std(-1), patch2image_3d_cross.std(-1))
    # rescale into value range of another modal
    patch2image_2d_cross_zero = patch2image_2d_cross.sum(-1) != 0
    patch2image_3d_cross = replace_sorted(patch2image_3d_org.reshape(-1), patch2image_3d_cross.reshape(-1)).reshape(patch_num_3d, ref_image_num)
    patch_no0_num = patch2image_2d_org[patch2image_2d_cross_zero].shape[0]
    patch2image_2d_org = patch2image_2d_org.float()
    patch2image_2d_cross = patch2image_2d_cross.float()
    tmp = replace_sorted(patch2image_2d_org[patch2image_2d_cross_zero].reshape(-1), patch2image_2d_cross[patch2image_2d_cross_zero].reshape(-1))
    patch2image_2d_cross[patch2image_2d_cross_zero] = tmp.reshape(patch_no0_num, ref_image_num)
    patch2image_2d_cross, _ = torch.cat((patch2image_2d_org.unsqueeze(-1), patch2image_2d_cross.unsqueeze(-1)), dim=-1).max(-1)
    patch2image_3d_cross, _ = torch.cat((patch2image_3d_org.unsqueeze(-1), patch2image_3d_cross.unsqueeze(-1)), dim=-1).max(-1)
    # anomaly enhancement
    patch2image_2d_cross *= confidence_2d.unsqueeze(-1)
    patch2image_3d_cross *= confidence_3d.unsqueeze(-1)
    patch2image_2d_cross = replace_sorted(patch2image_2d_org.reshape(-1), patch2image_2d_cross.reshape(-1))
    patch2image_2d_cross = patch2image_2d_cross.reshape(patch_num_2d, ref_image_num)
    patch2image_3d_cross = replace_sorted(patch2image_3d_org.reshape(-1), patch2image_3d_cross.reshape(-1))
    patch2image_3d_cross = patch2image_3d_cross.reshape(patch_num_3d, ref_image_num)
    patch2image_2d = (patch2image_2d_org + patch2image_2d_cross) / 2
    patch2image_3d = (patch2image_3d_org + patch2image_3d_cross) / 2
    # interval average
    k_max = max(1, int(ref_image_num*topmin_max))
    vals, _ = torch.topk(patch2image_2d.float(), k_max, largest=False, sorted=True)
    patch2image_2d = torch.mean(vals, dim=1)
    vals, _ = torch.topk(patch2image_3d.float(), k_max, largest=False, sorted=True)
    patch2image_3d = torch.mean(vals, dim=1)
    return patch2image_2d, patch2image_3d


def MSM_single(Z, device='cuda:0'):
    # Mutual scoring for only single modal
    anomaly_scores_matrix = torch.tensor([]).float().to(device)
    for i in tqdm(range(Z.shape[0])):
        anomaly_scores_i = compute_scores_single(Z, i).unsqueeze(0)   # (1, L)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.float()), dim=0)    # (N, L)
    return anomaly_scores_matrix


def MSM_multiple(Z_2d, Z_3d, device='cuda:0', patch2point_list=None, point2patch_list=None):
    # Mutual scoring for 2D+3D multimodal
    # cross-modal alignment
    anomaly_scores_2d_matrix = torch.tensor([]).float().to(device)
    anomaly_scores_3d_matrix = torch.tensor([]).float().to(device)
    for i in tqdm(range(Z_2d.shape[0])):
        anomaly_scores_i_2d_cross, anomaly_scores_i_3d_cross = compute_scores_multiple(Z_2d, Z_3d, i, patch2point_list=patch2point_list, point2patch_list=point2patch_list)   # (1, L)
        anomaly_scores_i_2d = anomaly_scores_i_2d_cross.unsqueeze(0)
        anomaly_scores_i_3d = anomaly_scores_i_3d_cross.unsqueeze(0)
        anomaly_scores_2d_matrix = torch.cat((anomaly_scores_2d_matrix, anomaly_scores_i_2d.float()), dim=0)    # (N, L)
        anomaly_scores_3d_matrix = torch.cat((anomaly_scores_3d_matrix, anomaly_scores_i_3d.float()), dim=0)    # (N, L)

    return anomaly_scores_2d_matrix, anomaly_scores_3d_matrix
