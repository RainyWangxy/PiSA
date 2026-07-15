import torch.nn.functional as F
import os
import argparse
import torch
from torch import nn
from lib.backbone import build_cnn_backbone, cnn_backbone_info, to_featuremap_backbone, GroupLinear

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from torchvision.models import ResNet50_Weights

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    
    def forward(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).squeeze(-1).squeeze(-1)


class ImageClassifier(torch.nn.Module):
    def __init__(self, P):
        super(ImageClassifier, self).__init__()
        
        self.arch = P['arch']
        feature_extractor = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-2])
        

        if P['freeze_feature_extractor']:
            for param in feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in feature_extractor.parameters():
                param.requires_grad = True
        self.feature_extractor = feature_extractor
            
        self.avgpool = GlobalAvgPool2d()

        linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
        self.linear_classifier = linear_classifier

    def forward(self, x):
        feats = self.feature_extractor(x)
        pooled_feats = self.avgpool(feats)
        logits = self.linear_classifier(pooled_feats)

        return logits

def build_model(backbone_name, num_classes, dim_embed, *, pretrained=True):

    class TransformerDecoderLayerWithoutSelfAttn(nn.Module):
        '''Similar to ML-Decoder
        '''

        def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True) -> None:
            super(TransformerDecoderLayerWithoutSelfAttn, self).__init__()
            self.norm1 = nn.LayerNorm(d_model, eps=1e-05)
            self.norm2 = nn.LayerNorm(d_model, eps=1e-05)
            self.norm3 = nn.LayerNorm(d_model, eps=1e-05)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
            self.activation = nn.ReLU()
            self.batch_first = batch_first

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                    memory_key_padding_mask=None):
            tgt = tgt + self.dropout1(tgt)
            tgt = self.norm1(tgt)
            tgt2, _ = self.multihead_attn(tgt, memory, memory)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt

    class MultilabelEncoderWithAttn(nn.Module):

        def __init__(self, num_classes, dim_feature, dim_embed):
            super().__init__()
            layer_decode = TransformerDecoderLayerWithoutSelfAttn(d_model=dim_embed, dim_feedforward=dim_embed * 4,
                                                                  dropout=0.1, nhead=8, batch_first=True)
            self.querys = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim_embed)
            self.requires_grad_(False)
            self.feature_projector = nn.Linear(dim_feature, dim_embed)
            self.query_projector = nn.Linear(dim_embed, dim_embed)
            self.decoder = nn.TransformerDecoder(layer_decode, num_layers=1)

        def forward(self, x):
            #  x: [batch_size, dim_feature, h, w]
            batch_size, _, h, w = x.shape

            x = x.flatten(2).transpose(1, 2)  # [batch_size, h*w, dim_feature]
            x = self.feature_projector(x)  # [batch_size, h*w, dim_embed]

            y = self.querys.weight  # [num_classes, dim_embed]
            y = self.query_projector(y)  # [num_classes, dim_embed]
            y = y.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_classes, dim_embed]

            representations = self.decoder(y, x)  # [batch_size, num_classes, dim_embed]
            return representations

    class MyModel(nn.Module):

        def __init__(self, backbone_name, num_classes, dim_embed=512, pretrained=False):
            super().__init__()
            # featuremap backbone (output: [batch_size, dim_feature, h, w])            
            cnn_backbone = build_cnn_backbone(name=backbone_name, pretrained=pretrained)
            dim_feature = cnn_backbone_info(cnn_backbone)['dim_featuremap']
            featuremap_backbone = to_featuremap_backbone(cnn_backbone)
            encoder = MultilabelEncoderWithAttn(num_classes, dim_feature=dim_feature, dim_embed=dim_embed)

            # for classification
            logit_projector = nn.Sequential(GroupLinear(num_classes, dim_embed, 1), nn.Flatten(1, 2))

            # for mutli-label prototype
            # prototype_embeds = F.normalize(torch.randn((num_classes, dim_embed)).float(), dim=-1)
            # prototype_embeds = nn.Parameter(prototype_embeds)
            # prototype_embeds.requires_grad_(True)
            self.backbone_name = backbone_name
            self.num_classes = num_classes
            self.dim_embed = dim_embed
            self.pretrained = pretrained
            self.dim_feature = dim_feature
            self.backbone = featuremap_backbone
            self.encoder = encoder
            self.logit_projector = logit_projector
            # self.prototype_embeds = prototype_embeds
            self.register_buffer("class_wise_prototypes", torch.zeros((self.num_classes, self.dim_embed),requires_grad=False))
            # self.register_buffer("class_wise_prototypes_ema", torch.zeros((self.num_classes, self.dim_embed),requires_grad=False))
        def forward(self, x):
            batch_size, _, H, W = x.shape  # [batch_size, 3, H, W]

            x = self.backbone(x)  # [batch_size, dim_feature, H2, W2]
            x = self.encoder(x)  # [batch_size, num_classer, dim_embed]

            # classification score
            clf_logits = self.logit_projector(x)  # [batch_size, num_classes, 1]
            x_norm = F.normalize(x, dim=2,p=2)  # [batch_size, num_classes, dim_embed]
            # prototype score
            proto = self.class_wise_prototypes
            proto = proto.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_classes, dim_embed]
            proto_score = F.cosine_similarity(x_norm, proto, dim=-1)  # [batch_size, num_classes]
            return clf_logits, proto_score.float(), x_norm

        def __repr__(self):
            return f'MyModel(backbone_name={self.backbone_name}, ' \
                   f'num_classes={self.num_classes}, ' \
                   f'dim_embed={self.dim_embed}, ' \
                   f'pretrained={self.pretrained})'
                   
        def update_prototypes(self, feature_w, label_vec_obs=None):
            # 更新原型
            feature_w = feature_w.clone().detach()
            batch_size, _, dim_embed = feature_w.shape
            for ins_idx in range(batch_size):
                for cls_idx in range(self.num_classes):
                    if label_vec_obs[ins_idx,cls_idx] == 1:
                        self.class_wise_prototypes[cls_idx,:] = self.class_wise_prototypes[cls_idx,:] + feature_w[ins_idx, cls_idx,:]
            self.class_wise_prototypes = F.normalize(self.class_wise_prototypes, dim=1)
    model = MyModel(backbone_name, num_classes, dim_embed=dim_embed, pretrained=pretrained)
    return model