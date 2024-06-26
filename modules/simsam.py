# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from modules import resnet

class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim, cluster_num=10):

        super(SimSiam, self).__init__()

        self.encoder = base_encoder
        self.instance_projector = nn.Sequential(
            nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim, bias=False),
            nn.BatchNorm1d(self.encoder.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder.rep_dim, dim)
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.rep_dim, cluster_num),
            nn.Softmax(dim=1)
        )

        self.instance_predictor = nn.Sequential(
            nn.Linear(dim, self.encoder.rep_dim, bias=False),
            nn.BatchNorm1d(self.encoder.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder.rep_dim, dim)
        )
        # self.cluster_predictor = nn.Sequential(
        #     nn.Linear(cluster_num, self.encoder.rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.encoder.rep_dim, cluster_num),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, x_i, x_j, x_s):

        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)
        z_s = self.encoder(x_s)

        p_instance_i = self.instance_projector(z_i)
        p_instance_j = self.instance_projector(z_j)
        p_instance_s = self.instance_projector(z_s)

        # c_cluster_i = self.cluster_projector(z_i)
        # c_cluster_j = self.cluster_projector(z_j)

        p_i = self.instance_predictor(p_instance_i)
        p_j = self.instance_predictor(p_instance_j)
        p_s = self.instance_predictor(p_instance_s)

        c_i = self.cluster_projector(z_i)
        c_j = self.cluster_projector(z_j)
        c_s = self.cluster_projector(z_s)



        return p_i, p_j, p_s, p_instance_i, p_instance_j, p_instance_s, c_i, c_j, c_s
    
    def forward_pui(self, x_i, x_j, x_s):
        z_i = self.encoder(x_i)
        z_j = self.encoder(x_j)
        z_s = self.encoder(x_s)

        c_i = self.cluster_projector(z_i)
        c_j = self.cluster_projector(z_j)
        c_s = self.cluster_projector(z_s)


        return c_i, c_j, c_s

    def forward_cluster(self, x):
        h = self.encoder(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)

        return c

    def forward_tsne(self, x):
        h = self.encoder(x)
        p = self.instance_projector(h)
        return p


# class SimSiam(nn.Module):
#     def __init__(self, base_encoder, dim, chao = 70,  cluster_num=10):
#         super(SimSiam, self).__init__()
#
#         self.encoder = base_encoder
#         self.instance_projector = nn.Sequential(
#             nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim, bias=False),
#             nn.BatchNorm1d(self.encoder.rep_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.encoder.rep_dim, dim)
#         )
#         self.cluster_projector = nn.Sequential(
#             nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.encoder.rep_dim, chao),
#             nn.Softmax(dim=1)
#         )
#
#         self.instance_predictor = nn.Sequential(
#             nn.Linear(dim, self.encoder.rep_dim, bias=False),
#             nn.BatchNorm1d(self.encoder.rep_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.encoder.rep_dim, dim)
#         )
#         self.cluster_predictor = nn.Sequential(
#             nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim),
#             nn.ReLU(),
#             nn.Linear(self.encoder.rep_dim, cluster_num),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x_i, x_j, x_s):
#         z_i = self.encoder(x_i)
#         z_j = self.encoder(x_j)
#         z_s = self.encoder(x_s)
#
#         p_instance_i = self.instance_projector(z_i)
#         p_instance_j = self.instance_projector(z_j)
#         p_instance_s = self.instance_projector(z_s)
#
#         c_cluster_i = self.cluster_projector(z_i)
#         c_cluster_j = self.cluster_projector(z_j)
#         c_cluster_s = self.cluster_projector(z_j)
#
#         p_i = self.instance_predictor(p_instance_i)
#         p_j = self.instance_predictor(p_instance_j)
#         p_s = self.instance_predictor(p_instance_s)
#
#         c_i = self.cluster_predictor(z_i)
#         c_j = self.cluster_predictor(z_j)
#         c_s = self.cluster_predictor(z_s)
#
#         return p_i, p_j, p_s, p_instance_i, p_instance_j, p_instance_s, c_i, c_j, c_s, c_cluster_i, c_cluster_j, c_cluster_s
#
#     def forward_cluster(self, x):
#         h = self.encoder(x)
#         c = self.cluster_predictor(h)
#         c = torch.argmax(c, dim=1)
#
#         return c
#
#     def forward_tsne(self, x):
#         h = self.encoder(x)
#         p = self.instance_projector(h)
#         return p