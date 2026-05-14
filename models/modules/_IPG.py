import torch
import numpy as np

def closed_k(xyz, support_idx, query_idx, k):
    mask = ~torch.isin(support_idx, query_idx.view(-1))
    diff_tensor = support_idx[mask]
    support_idx = torch.unique(diff_tensor)
    support_xyz = xyz[support_idx]
    query_xyz = xyz[query_idx]
    distance = torch.cdist(query_xyz, support_xyz)
    _, indices = torch.topk(distance.reshape(-1), k, largest=False)
    indices = indices % support_idx.shape[0]
    indices = support_idx[indices]
    return indices


def iter_group(xyz, center_idx, idx_all, group_size):
    K_iter = 80
    Cthr = 0.01
    Nc = center_idx.shape[1]
    xyz, center_idx, idx_all = xyz.squeeze(), center_idx.squeeze(), idx_all.squeeze()
    iter_idx = [None] * Nc
    
    # compute curvatures
    center_xyz = xyz[center_idx, :]
    neighbors_xyz = xyz[idx_all[center_idx, :group_size]]
    neighbors_xyz = neighbors_xyz[:, 1:, :] - center_xyz.unsqueeze(1)
    neighbors_xyz /= neighbors_xyz.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0]
    neighbors_xyz = neighbors_xyz.permute(0, 2, 1)
    neighbors_xyz_mean = neighbors_xyz.mean(-1, keepdims=True)
    tmpa = neighbors_xyz - neighbors_xyz_mean
    covariance_matrix = torch.bmm(tmpa, tmpa.permute(0, 2, 1))/(group_size-2)   # (Nc, 3, 3)
    eigenvalues = np.linalg.eig(covariance_matrix.cpu().float())[0]
    curvatures = eigenvalues.min(-1) / eigenvalues.sum(-1)

    # group initialization
    iter_idx = [idx_all[center_idx[i], :group_size] if curvatures[i] < Cthr else idx_all[center_idx[i], :K_iter] for i in range(Nc)]
    # group expansion
    search_space_idx = idx_all[center_idx, :]
    for i in range(Nc):
        while True:
            if iter_idx[i].shape[0] == group_size:
                break
            if iter_idx[i].shape[0] + K_iter <= group_size:
                iter_idx_i = closed_k(xyz, search_space_idx[i], iter_idx[i], K_iter)
            else:
                last_num = group_size - iter_idx[i].shape[0]
                iter_idx_i = closed_k(xyz, search_space_idx[i], iter_idx[i], last_num)
            iter_idx[i] = torch.cat([iter_idx_i, iter_idx[i]], dim=0)

    iter_idx = torch.stack(iter_idx, dim=0).unsqueeze(0)
    iter_idx_ = iter_idx.view(-1)
    center = xyz[center_idx, :]
    neighborhood = xyz.squeeze()[iter_idx_, :]
    neighborhood = neighborhood.reshape(1, Nc, group_size, 3).contiguous()
    neighborhood = neighborhood - center.unsqueeze(0).unsqueeze(2)  # (1, 1024, 128, 3)
    
    return iter_idx, curvatures, neighborhood