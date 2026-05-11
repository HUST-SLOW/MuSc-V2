"""
    The PatchMaker is copied from https://github.com/amazon-science/patchcore-inspection.
"""

import torch
import torch.nn.functional as F
import math

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class SNAMD(torch.nn.Module):
    def __init__(self, device, r_list=[1,3,5]):
        super(SNAMD, self).__init__()
        self.device = device
        self.r_list = r_list
        self.r_max = max(r_list)
        self.patch_maker = PatchMaker(self.r_max, stride=1)
    
    def _embed_2d(self, features):
        B = features[0].shape[0]
        features_layers = []
        for feature in features:
            feature = feature[:, 1:, :].to(self.device)
            feature = feature.reshape(feature.shape[0],
                                        int(math.sqrt(feature.shape[1])),
                                        int(math.sqrt(feature.shape[1])),
                                        feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)
            features_layers.append(feature)

        if self.r_max != 1:
            # divide into patches
            features_r_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
            features_r_layers = [x[0] for x in features_r_layers]
        else:
            features_r_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in features_layers]
        features_r_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_r_layers]
        
        # Similarity-Weighted Pooling
        layer_num = len(features_r_layers)
        features_layers_list = torch.tensor([]).half().to(features_r_layers[0].device)
        for i in range(layer_num):
            center = self.r_max // 2
            features_layers_ = features_r_layers[i]
            # compute the similarity matrix
            center_feature = features_layers_[:, :, center:center+1, center:center+1]
            dist_matrix = (features_layers_ - center_feature).norm(dim=1, keepdim=True)
            weight_matrix = torch.exp(-dist_matrix)    # (B*L, 1, r, r)
            shift_value = [(r_-1)//2 for r_ in self.r_list]
            left = [(self.r_max-2*st-1)//2 for st in shift_value]
            right = [self.r_max-(self.r_max-2*st-1)//2 for st in shift_value]
            features_layers_ = [features_layers_[:, :, left[r_i]:right[r_i], left[r_i]:right[r_i]] for r_i in range(len(self.r_list))]
            weight_matrix_ = [weight_matrix[:, :, left[r_i]:right[r_i], left[r_i]:right[r_i]] for r_i in range(len(self.r_list))]
            # weighted average
            features_layers__ = []
            for f_i, f in enumerate(features_layers_):
                weight_matrix = weight_matrix_[f_i]
                features_layers = f * weight_matrix
                weight_matrix_sum = weight_matrix.reshape(weight_matrix.shape[0], -1).sum(-1, keepdim=True).unsqueeze(-1)
                features_layers = features_layers.reshape(*features_layers.shape[:2], -1).sum(dim=-1, keepdim=True)
                features_layers = torch.div(features_layers, weight_matrix_sum.repeat(1, features_layers.shape[1], 1)).half()
                features_layers__.append(features_layers)
            features_layers = torch.cat(features_layers__, dim=-1).mean(dim=-1, keepdim=True)
            features_layers_list = torch.cat((features_layers_list, features_layers), dim=-1)
        features_layers = features_layers_list.reshape(B, -1, *features_layers_list.shape[-2:]).permute(0, 1, 3, 2)   # (B, L, layer, C)
        return features_layers


    def _embed_3d(self, features, center, curvatures):
        # (B, N, C) (B, 3, N) (B, N)
        B, _, C = features[0].shape
        features_layers = [f.reshape(-1, C) for f in features]
        center = center.permute(0, 2, 1).reshape(-1, 3)

        if self.r_max != 1:
            # search neighborhood by center
            point2center = torch.cdist(center, center)
            _, idx = torch.topk(point2center.float(), self.r_max, largest=False, sorted=True)
            for i in range(len(features_layers)):
                features_neighbors = features_layers[i][idx.reshape(-1)].reshape(-1, self.r_max, C)
                features_neighbors = features_neighbors.permute(0, 2, 1)
                features_layers[i] = features_neighbors # (N, C, r)
        else:
            for i in range(len(features_layers)):
                features_neighbors = features_layers[i].reshape(-1, self.r_max, C)
                features_neighbors = features_neighbors.permute(0, 2, 1)
                features_layers[i] = features_neighbors
        
        # Similarity-Weighted Pooling
        layer_num = len(features_layers)
        features_layers_list = torch.tensor([]).half().to(features_layers[0].device)
        for i in range(layer_num):
            features_layer = features_layers[i]
            # compute the similarity matrix
            center_feature = features_layer[:, :, 0:1]
            dist_matrix = (features_layer - center_feature).norm(dim=1, keepdim=True)
            weight_matrix = torch.exp(-dist_matrix)    # (B*L, 1, r)
            features_layer_r = [features_layer[:, :, :r_3d] for r_3d in self.r_list]
            weight_matrix_ = [weight_matrix[:, :, :r_3d] for r_3d in self.r_list]
            # weighted average
            features_layers__ = []
            for f_i, f in enumerate(features_layer_r):
                weight_matrix = weight_matrix_[f_i]
                features_layers_ = f * weight_matrix
                weight_matrix_sum = weight_matrix.reshape(weight_matrix.shape[0], -1).sum(-1, keepdim=True).unsqueeze(-1)
                features_layers_ = features_layers_.reshape(*features_layers_.shape[:2], -1).sum(dim=-1, keepdim=True)
                features_layers_ = torch.div(features_layers_, weight_matrix_sum.repeat(1, features_layers_.shape[1], 1)).half()
                features_layers__.append(features_layers_)
            features_layers_ = torch.cat(features_layers__, dim=-1)
            # surface-consistent aggregation
            agg_degree_num = features_layers_.shape[-1]
            features_layers_high = center_feature[:, :, :].squeeze().half()
            features_layers_low = features_layers_.permute(0, 2, 1).reshape(features_layers_.shape[0], -1)
            features_layers_low = F.adaptive_avg_pool1d(features_layers_low, int(features_layers_low.shape[-1]/agg_degree_num))  # 降维
            features_layers_low = features_layers_low.squeeze().half()
            features_layers_ = torch.zeros_like(features_layers_low)
            Cthr = 0.01
            features_layers_[curvatures < Cthr] = features_layers_low[curvatures < Cthr]
            features_layers_[curvatures >= Cthr] = features_layers_high[curvatures >= Cthr]
            features_layers_ = features_layers_.unsqueeze(-1)
            features_layers_list = torch.cat((features_layers_list, features_layers_.half()), dim=-1)
        features_layers = features_layers_list.reshape(B, -1, *features_layers_list.shape[-2:]).permute(0, 1, 3, 2)   # (B, L, layer, C)
        return features_layers

