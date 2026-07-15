import numpy as np
import torch
import torch.nn as nn

LOG_EPSILON = 1e-5
def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def loss_an(preds, label_vec_obs):
    observed_labels = label_vec_obs
    assert torch.min(observed_labels) >= 0
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    return loss_mtx

def LargeLossMattersMask(score, label_vec_obs, loss_mtx, ignore_num=1):
    pseudo_labels = torch.zeros_like(label_vec_obs)
    for ins_idx in range(score.shape[0]):
        sample_score = score[ins_idx]
        masked_score = torch.where(label_vec_obs == 0, sample_score, -np.inf) 
        high_sim_cls = torch.topk(masked_score, ignore_num).indices
        pseudo_labels[ins_idx,high_sim_cls] = 1
    mask = (pseudo_labels == 1)
    loss_mtx[mask] = 0 
    return loss_mtx

class PrototypeLoss(nn.Module):
    def __init__(self, gamma):
        super(PrototypeLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features_w, class_wise_prototypes, label_vec_obs):
        """remove features_s"""
        batch_size, num_classes,_ = features_w.shape
        mask_pos = torch.zeros((batch_size,num_classes, num_classes)).cuda()
        mask_neg = torch.zeros((batch_size,num_classes, num_classes)).cuda()
        for i in range(batch_size):
            for cls_idx in range(num_classes):
                if label_vec_obs[i, cls_idx] == 1:
                    mask_pos[i, cls_idx, cls_idx] = 1
                    mask_neg[i, cls_idx, :] = 1
                    mask_neg[i, cls_idx, cls_idx] = 0
        dot_prod_w = torch.matmul(features_w, class_wise_prototypes.transpose(-1, -2))
        pos_pairs_mean_w = (mask_pos * dot_prod_w).sum(dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_w = torch.abs(mask_neg * dot_prod_w).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)
        loss_w = ((1.0 - pos_pairs_mean_w) + (self.gamma * neg_pairs_mean_w)).mean()

        return loss_w
def PGRL(score_proto, label_vec_obs, loss_mtx, ignore_num=5):
    # ignore_num = int(left_rate*score_proto.shape[1]) + 1
    pseudo_labels = torch.zeros_like(label_vec_obs)
    for ins_idx in range(score_proto.shape[0]):
        sample_proto = score_proto[ins_idx]
        high_sim_cls = torch.topk(sample_proto, ignore_num).indices
        pseudo_labels[ins_idx,high_sim_cls] = 1
    mask = ((label_vec_obs == 0) & (pseudo_labels == 1))
    loss_mtx[mask] = 0 
    return loss_mtx, mask